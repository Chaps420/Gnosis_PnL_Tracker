from flask import Flask, request, jsonify
from flask_cors import CORS
import xrpl
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountTx, AccountLines, BookOffers, AMMInfo
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
import requests
import binascii
import cachetools
import asyncio
import aiohttp
import concurrent.futures
import time
from decimal import Decimal
import contextlib
import signal
from functools import wraps

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# XRPL client setup
JSON_RPC_URL = "https://s1.ripple.com:51234/"
client = JsonRpcClient(JSON_RPC_URL)

# DEX Screener API setup
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex/search"

# Cache setup
CACHE_TTL = 300  # 5 minutes
price_cache = cachetools.TTLCache(maxsize=500, ttl=CACHE_TTL)
amm_cache = cachetools.TTLCache(maxsize=50, ttl=CACHE_TTL)

# Thread pool for XRPL requests
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Timeout decorator
def timeout(seconds):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

def decode_hex_currency(hex_code):
    """Decode a 40-character hex currency code to ASCII."""
    if len(hex_code) == 40 and all(c in '0123456789ABCDEFabcdef' for c in hex_code):
        try:
            bytes_code = binascii.unhexlify(hex_code)
            return bytes_code.split(b'\0', 1)[0].decode('ascii')
        except Exception:
            pass
    return hex_code

def decode_currency(currency, issuer, amm_info_cache=None):
    """Decode currency and identify AMM LP tokens using AMMInfo."""
    if len(currency) == 40 and all(c in '0123456789ABCDEFabcdef' for c in currency):
        if amm_info_cache and issuer in amm_info_cache:
            amm_info = amm_info_cache[issuer]
        else:
            cache_key = f"amm_{issuer}"
            if cache_key in amm_cache:
                amm_info = amm_cache[cache_key]
            else:
                try:
                    amm_info = client.request(AMMInfo(amm_account=issuer)).result["amm"]
                    amm_cache[cache_key] = amm_info
                except Exception as e:
                    logger.debug(f"AMMInfo failed for {issuer}: {e}")
                    return decode_hex_currency(currency), False
        required_fields = ["lp_token", "amount", "amount2"]
        if all(field in amm_info for field in required_fields):
            asset1 = amm_info["amount"]
            asset2 = amm_info["amount2"]
            asset1_str = "XRP" if isinstance(asset1, str) else decode_hex_currency(asset1["currency"])
            asset2_str = "XRP" if isinstance(asset2, str) else decode_hex_currency(asset2["currency"])
            return f"LP_{asset1_str}_{asset2_str}", True
        return decode_hex_currency(currency), False
    return currency, False

def get_balance_changes(meta, address, relevant_tokens):
    """Extract balance changes for relevant tokens."""
    changes = {'XRP': 0}
    for node in meta.get('AffectedNodes', []):
        if 'ModifiedNode' in node:
            modified = node['ModifiedNode']
            if modified.get('LedgerEntryType') == 'AccountRoot' and 'PreviousFields' in modified:
                final_fields = modified.get('FinalFields', {})
                previous_fields = modified.get('PreviousFields', {})
                if final_fields.get('Account') == address and 'Balance' in previous_fields:
                    final_balance = int(final_fields.get('Balance', 0))
                    previous_balance = int(previous_fields['Balance'])
                    changes['XRP'] += final_balance - previous_balance
            elif modified.get('LedgerEntryType') == 'RippleState' and 'PreviousFields' in modified:
                final_fields = modified.get('FinalFields', {})
                previous_fields = modified.get('PreviousFields', {})
                if 'Balance' in previous_fields:
                    high = final_fields.get('HighLimit', {}).get('issuer')
                    low = final_fields.get('LowLimit', {}).get('issuer')
                    if high == address or low == address:
                        currency = final_fields.get('Balance', {}).get('currency')
                        issuer = high if low == address else low
                        token_key = f"{currency}-{issuer}"
                        if token_key in relevant_tokens:
                            final_balance = float(final_fields.get('Balance', {}).get('value', 0))
                            previous_balance = float(previous_fields['Balance']['value'])
                            delta = final_balance - previous_balance
                            changes[token_key] = changes.get(token_key, 0) + (delta if low == address else -delta)
        elif 'CreatedNode' in node:
            created = node['CreatedNode']
            if created.get('LedgerEntryType') == 'RippleState':
                new_fields = created.get('NewFields', {})
                high = new_fields.get('HighLimit', {}).get('issuer')
                low = new_fields.get('LowLimit', {}).get('issuer')
                if high == address or low == address:
                    currency = new_fields.get('Balance', {}).get('currency')
                    issuer = high if low == address else low
                    token_key = f"{currency}-{issuer}"
                    if token_key in relevant_tokens:
                        balance = float(new_fields.get('Balance', {}).get('value', 0))
                        changes[token_key] = changes.get(token_key, 0) + (balance if low == address else -balance)
        elif 'DeletedNode' in node:
            deleted = node['DeletedNode']
            if deleted.get('LedgerEntryType') == 'RippleState':
                final_fields = deleted.get('FinalFields', {})
                high = final_fields.get('HighLimit', {}).get('issuer')
                low = final_fields.get('LowLimit', {}).get('issuer')
                if high == address or low == address:
                    currency = final_fields.get('Balance', {}).get('currency')
                    issuer = high if low == address else low
                    token_key = f"{currency}-{issuer}"
                    if token_key in relevant_tokens:
                        balance = float(final_fields.get('Balance', {}).get('value', 0))
                        changes[token_key] = changes.get(token_key, 0) + (-balance if low == address else balance)
    return changes

async def fetch_dexscreener_batch_async(session, token_ids, semaphore):
    """Fetch prices for a batch of tokens from DEX Screener API asynchronously."""
    start_time = time.time()
    async with semaphore:
        try:
            url = f"{DEXSCREENER_API_URL}?q={','.join(token_ids)}"
            async with session.get(url, timeout=5) as response:
                data = await response.json()
                prices = {}
                for pair in data.get('pairs', []):
                    if pair.get('chainId') == 'xrpl' and pair.get('quoteToken', {}).get('symbol') == 'XRP':
                        base_token = pair.get('baseToken', {}).get('symbol')
                        issuer = pair.get('baseToken', {}).get('address')
                        price = float(pair.get('priceNative', '0'))
                        prices[f"{base_token}-{issuer}"] = price
                logger.debug(f"DEX Screener batch fetched {len(prices)} prices in {time.time() - start_time:.2f}s")
                return prices
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            logger.error(f"DEX Screener batch error: {e} in {time.time() - start_time:.2f}s")
            return {}

async def batch_fetch_dexscreener_prices(currency_issuer_pairs):
    """Fetch DEX Screener prices in batches of 30 tokens with limited concurrency."""
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent batch requests
    tasks = []
    for i in range(0, len(currency_issuer_pairs), 30):
        batch = currency_issuer_pairs[i:i+30]
        token_ids = [f"{decode_hex_currency(currency)}.{issuer}" for currency, issuer in batch]
        tasks.append(fetch_dexscreener_batch_async(aiohttp.ClientSession(), token_ids, semaphore))
    batch_results = await asyncio.gather(*tasks)
    prices = {}
    for batch_result in batch_results:
        prices.update(batch_result)
    return [prices.get(f"{decode_hex_currency(currency)}.{issuer}", None) for currency, issuer in currency_issuer_pairs]

def get_dexscreener_price_sync(decoded_currency, issuer):
    """Synchronous DEX Screener fallback using requests."""
    cache_key = f"dexscreener_{decoded_currency}_{issuer}"
    if cache_key in price_cache:
        return price_cache[cache_key]
    
    try:
        url = f"{DEXSCREENER_API_URL}?q={decoded_currency}.{issuer}"
        response = requests.get(url, timeout=3).json()
        for pair in response.get('pairs', []):
            if pair.get('chainId') == 'xrpl' and pair.get('baseToken', {}).get('symbol') == decoded_currency and pair.get('quoteToken', {}).get('symbol') == 'XRP':
                price = float(pair.get('priceNative', '0'))
                price_cache[cache_key] = price
                return price
        price_cache[cache_key] = None
        return None
    except (requests.RequestException, ValueError) as e:
        logger.error(f"DEX Screener sync error for {decoded_currency}/{issuer}: {e}")
        price_cache[cache_key] = None
        return None

def get_dexscreener_price(decoded_currency, issuer, batch_results=None, batch_index=None):
    """Get DEX Screener price, using batch results or sync fallback."""
    cache_key = f"dexscreener_{decoded_currency}_{issuer}"
    if cache_key in price_cache:
        return price_cache[cache_key]
    
    if batch_results and batch_index is not None:
        price = batch_results[batch_index]
        if isinstance(price, Exception):
            price = None
        price_cache[cache_key] = price
        return price
    
    return get_dexscreener_price_sync(decoded_currency, issuer)

def get_dex_price(currency, issuer):
    """Fetch price from XRPL DEX."""
    cache_key = f"dex_{currency}_{issuer}"
    if cache_key in price_cache:
        return price_cache[cache_key]
    
    def fetch_offers(request):
        try:
            return client.request(request).result.get("offers", [])
        except Exception:
            return []
    
    try:
        buy_request = BookOffers(taker_pays={"currency": "XRP"}, taker_gets={"currency": currency, "issuer": issuer}, limit=3)
        sell_request = BookOffers(taker_gets={"currency": "XRP"}, taker_pays={"currency": currency, "issuer": issuer}, limit=3)
        future_buy = executor.submit(fetch_offers, buy_request)
        future_sell = executor.submit(fetch_offers, sell_request)
        buy_offers = future_buy.result()
        sell_offers = future_sell.result()

        MIN_OFFERS, MIN_VOLUME = 1, 1000
        buy_price = None
        if buy_offers:
            xrp = sum(float(o["TakerPays"]) / 1_000_000 for o in buy_offers)
            tokens = sum(float(o["TakerGets"]["value"]) for o in buy_offers)
            if tokens >= MIN_VOLUME:
                buy_price = xrp / tokens
        sell_price = None
        if sell_offers:
            xrp = sum(float(o["TakerGets"]) / 1_000_000 for o in sell_offers)
            tokens = sum(float(o["TakerPays"]["value"]) for o in sell_offers)
            if tokens >= MIN_VOLUME:
                sell_price = xrp / tokens
        if buy_price and sell_price:
            price = (buy_price + sell_price) / 2
        else:
            price = buy_price or sell_price
        if price:
            price_cache[cache_key] = price
            return price
        price_cache[cache_key] = None
        return None
    except Exception as e:
        logger.error(f"XRPL DEX price error for {currency}-{issuer}: {e}")
        price_cache[cache_key] = None
        return None

def get_historical_price(decoded_currency, issuer, transactions):
    """Fetch price from historical transactions within last 60 days."""
    cache_key = f"historical_{decoded_currency}_{issuer}"
    if cache_key in price_cache:
        return price_cache[cache_key]
    
    prices = []
    cutoff = datetime.utcnow() - timedelta(days=60)
    for tx in transactions:
        tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
        if tx_time < cutoff or not tx.get('meta'):
            continue
        meta = tx["meta"]
        if (isinstance(meta.get("delivered_amount"), dict) and 
            meta["delivered_amount"].get("currency") == "XRP" and
            tx.get("tx", {}).get("TransactionType") in ["Payment", "OfferCreate"]):
            xrp = float(meta["delivered_amount"]["value"]) / 1_000_000
            for node in meta.get("AffectedNodes", []):
                if ("ModifiedNode" in node and 
                    node["ModifiedNode"].get("LedgerEntryType") == "RippleState" and 
                    node["ModifiedNode"].get("FinalFields", {}).get("Balance", {}).get("currency") == decoded_currency and 
                    node["ModifiedNode"].get("HighLimit", {}).get("issuer") == issuer):
                    token = float(node["ModifiedNode"]["FinalFields"]["Balance"]["value"])
                    if token != 0:
                        prices.append(xrp / abs(token))
    price = sum(prices) / len(prices) if prices else None
    if price:
        price_cache[cache_key] = price
        return price
    price_cache[cache_key] = None
    return None

def get_current_price(currency, issuer, transactions, batch_results=None, batch_index=None):
    """Fetch current token price in XRP with fallbacks."""
    cache_key = f"current_{currency}_{issuer}"
    if cache_key in price_cache:
        return price_cache[cache_key]
    
    decoded_currency = decode_hex_currency(currency)
    for method in (
        lambda: get_dexscreener_price(decoded_currency, issuer, batch_results, batch_index),
        lambda: get_dex_price(currency, issuer),
        lambda: get_historical_price(decoded_currency, issuer, transactions)
    ):
        try:
            price = method()
            if price and price > 0:
                price_cache[cache_key] = price
                return price
        except Exception as e:
            logger.error(f"Price method failed for {decoded_currency}/{issuer}: {e}")
            continue
    price_cache[cache_key] = None
    return None

def get_lp_token_value(issuer, amount_held, transactions, amm_info_cache=None):
    """Calculate LP token value based on AMM pool data."""
    try:
        cache_key = f"amm_{issuer}"
        if amm_info_cache and issuer in amm_info_cache:
            amm_info = amm_info_cache[issuer]
        elif cache_key in amm_cache:
            amm_info = amm_cache[cache_key]
        else:
            amm_info = client.request(AMMInfo(amm_account=issuer)).result["amm"]
            amm_cache[cache_key] = amm_info

        lp_tokens_issued = float(amm_info["lp_token"]["value"])
        asset1 = amm_info["amount"]
        asset2 = amm_info["amount2"]

        if isinstance(asset1, str):
            amount_xrp = float(asset1) / 1_000_000
            token_currency = asset2["currency"]
            token_issuer = asset2["issuer"]
            amount_token = float(asset2["value"])
        elif isinstance(asset2, str):
            amount_xrp = float(asset2) / 1_000_000
            token_currency = asset1["currency"]
            token_issuer = asset1["issuer"]
            amount_token = float(asset1["value"])
        else:
            logger.warning(f"Both assets are tokens for {issuer}, not supported")
            return None

        token_price = get_current_price(token_currency, token_issuer, transactions)
        if token_price is None:
            logger.warning(f"No price for LP token asset {token_currency}/{token_issuer}")
            return None
        total_pool_value = amount_xrp + (amount_token * token_price)
        value_per_lp = total_pool_value / lp_tokens_issued
        current_value = amount_held * value_per_lp
        return current_value
    except Exception as e:
        logger.error(f"Error calculating LP token value for {issuer}: {e}")
        return None

@app.route('/token_pnl', methods=['POST'])
@timeout(150)
def get_token_pnl():
    start_time = time.time()
    data = request.json
    address = data.get('address')

    if not address:
        return jsonify({'error': 'Wallet address is required'}), 400

    transactions = []
    token_changes = defaultdict(list)

    try:
        # Fetch current holdings
        req = AccountLines(account=address)
        response = client.request(req)
        lines = response.result['lines']
        holdings = {f"{line['currency']}-{line['account']}": float(line['balance']) for line in lines if float(line['balance']) > 0.0001}
        relevant_tokens = set(holdings.keys())

        # Check for $UGA or $GNOSIS
        has_uga_or_gnosis = any(token in holdings for token in ["UGA-rBFJGmWj6YaabVCxfsjiCM8pfYXs8xFdeC", "474E4F5349530000000000000000000000000000-rHUQ3xYC2hwfJa9idjjmsCcb5hP3qZiiTM"])
        if not has_uga_or_gnosis:
            return jsonify({
                'error': 'This wallet does not contain $UGA or $GNOSIS - Go to the following to purchase -',
                'purchase_links': [
                    'https://firstledger.net/token/rHUQ3xYC2hwfJa9idjjmsCcb5hP3qZiiTM/474E4F5349530000000000000000000000000000',
                    'https://firstledger.net/token/rBFJGmWj6YaabVCxfsjiCM8pfYXs8xFdeC/UGA'
                ]
            }), 400

        # Pre-fetch AMM info
        amm_info_cache = {}
        for token in holdings:
            _, issuer = token.split('-')
            if issuer not in amm_info_cache:
                try:
                    amm_info = client.request(AMMInfo(amm_account=issuer)).result["amm"]
                    amm_info_cache[issuer] = amm_info
                except Exception:
                    pass

        # Fetch transactions
        marker = None
        tx_count = 0
        cutoff_time = datetime.utcnow() - timedelta(days=60)
        while tx_count < 1000:
            req = AccountTx(account=address, ledger_index_min=-1, ledger_index_max=-1, limit=200, marker=marker, forward=True)
            response = client.request(req)
            result = response.result
            new_txs = result.get('transactions', [])
            for tx in new_txs:
                tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
                if tx_time < cutoff_time:
                    continue
                transactions.append(tx)
                tx_count += 1
                if tx_count >= 1000:
                    break
            marker = result.get('marker')
            if not marker or tx_count >= 1000:
                break

        # Precompute balance changes for all transactions
        for tx in transactions:
            meta = tx.get('meta', {})
            if isinstance(meta, dict):
                changes = get_balance_changes(meta, address, relevant_tokens)
                delta_xrp = changes.get('XRP', 0) / 1_000_000
                for token in relevant_tokens:
                    delta_token = changes.get(token, 0)
                    token_changes[token].append((delta_xrp, delta_token))

        # Batch fetch DEX Screener prices
        currency_issuer_pairs = [(decode_hex_currency(token.split('-')[0]), token.split('-')[1]) for token in holdings]
        batch_results = asyncio.run(batch_fetch_dexscreener_prices(currency_issuer_pairs))

        regular_tokens = []
        amm_lp_tokens = []
        for i, token in enumerate(holdings):
            currency, issuer = token.split('-')
            currency_name, is_amm_lp = decode_currency(currency, issuer, amm_info_cache)
            buys = deque()
            realized_pnl = 0.0

            for delta_xrp, delta_token in token_changes[token]:
                if delta_xrp < 0 and delta_token > 0:  # Buy
                    price = -delta_xrp / delta_token
                    buys.append({'amount': delta_token, 'price': price})
                elif delta_xrp > 0 and delta_token < 0:  # Sell
                    sell_amount = -delta_token
                    sell_value = delta_xrp
                    while sell_amount > 0 and buys:
                        buy = buys[0]
                        if buy['amount'] <= sell_amount:
                            realized_pnl += (sell_value / sell_amount - buy['price']) * buy['amount']
                            sell_amount -= buy['amount']
                            buys.popleft()
                        else:
                            realized_pnl += (sell_value / sell_amount - buy['price']) * sell_amount
                            buy['amount'] -= sell_amount
                            sell_amount = 0

            cost_basis = sum(buy['amount'] * buy['price'] for buy in buys)
            batch_index = i
            current_price = get_current_price(currency, issuer, transactions, batch_results, batch_index)
            current_value = None
            if is_amm_lp:
                current_value = get_lp_token_value(issuer, holdings[token], transactions, amm_info_cache)
            elif current_price is not None:
                current_value = float(Decimal(str(holdings[token])) * Decimal(str(current_price)))
            unrealized_pnl = current_value - cost_basis if current_value is not None else None
            total_pnl = realized_pnl + unrealized_pnl if unrealized_pnl is not None else None

            token_data = {
                'currency': currency_name,
                'issuer': issuer,
                'amount_held': holdings[token],
                'initial_investment': cost_basis,
                'current_value': current_value,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl
            }

            if is_amm_lp:
                amm_lp_tokens.append(token_data)
            else:
                regular_tokens.append(token_data)

        regular_tokens.sort(key=lambda x: x['current_value'] if x['current_value'] is not None else 0, reverse=True)
        amm_lp_tokens.sort(key=lambda x: x['current_value'] if x['current_value'] is not None else 0, reverse=True)

        return jsonify({
            'xrp_balance': sum(float(line['balance']) for line in lines if line['currency'] == 'XRP'),
            'tokens': regular_tokens,
            'amm_lp_tokens': amm_lp_tokens
        })
    except TimeoutError:
        logger.error(f"Request timeout for {address} after {time.time() - start_time:.2f}s")
        return jsonify({
            'error': 'Request took too long to process. Try a wallet with fewer transactions or try again later.'
        }), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        transactions.clear()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
