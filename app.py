from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountTx, AccountLines, BookOffers, AMMInfo
from datetime import datetime, timedelta
from collections import deque
import logging
import aiohttp
import asyncio
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
import binascii

app = Flask(__name__)
CORS(app)

# Rate limiting to prevent abuse
limiter = Limiter(app, key_func=get_remote_address, default_limits=["10 per minute"])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# XRPL client setup
JSON_RPC_URL = "https://s1.ripple.com:51234/"
client = JsonRpcClient(JSON_RPC_URL)

# DEX Screener API setup
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex/search"

# In-memory caches (TTL: 5 minutes)
price_cache = TTLCache(maxsize=1000, ttl=300)
amm_cache = TTLCache(maxsize=100, ttl=300)
pnl_cache = TTLCache(maxsize=100, ttl=300)

def decode_hex_currency(hex_code):
    """Decode a 40-character hex currency code to ASCII."""
    if len(hex_code) == 40 and all(c in '0123456789ABCDEFabcdef' for c in hex_code):
        try:
            bytes_code = binascii.unhexlify(hex_code)
            return bytes_code.split(b'\0', 1)[0].decode('ascii')
        except Exception:
            pass
    return hex_code

@lru_cache(maxsize=1000)
def decode_currency(currency, issuer):
    """Decode currency and identify AMM LP tokens using AMMInfo."""
    if len(currency) == 40 and all(c in '0123456789ABCDEFabcdef' for c in currency):
        try:
            cache_key = f"amm:{issuer}"
            if cache_key in amm_cache:
                amm_info = amm_cache[cache_key]
            else:
                amm_info = client.request(AMMInfo(amm_account=issuer)).result["amm"]
                amm_cache[cache_key] = amm_info
            required_fields = ["lp_token", "amount", "amount2"]
            if all(field in amm_info for field in required_fields):
                asset1 = amm_info["amount"]
                asset2 = amm_info["amount2"]
                asset1_str = "XRP" if isinstance(asset1, str) else decode_hex_currency(asset1["currency"])
                asset2_str = "XRP" if isinstance(asset2, str) else decode_hex_currency(asset2["currency"])
                return f"LP_{asset1_str}_{asset2_str}", True
            return decode_hex_currency(currency), False
        except Exception as e:
            logger.debug(f"AMMInfo failed for {issuer}: {e}")
            return decode_hex_currency(currency), False
    return currency, False

def get_balance_changes(meta, address):
    """Extract balance changes from transaction metadata."""
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
                        final_balance = float(final_fields.get('Balance', {}).get('value', 0))
                        previous_balance = float(previous_fields['Balance']['value'])
                        delta = final_balance - previous_balance
                        changes[token_key] = changes.get(token_key, 0) + (delta if low == address else -delta)
        # Simplified: Omitted CreatedNode/DeletedNode for brevity; add back if needed
    return changes

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
async def fetch_dexscreener_price(decoded_currency):
    """Fetch price from DEX Screener with retry logic."""
    async with aiohttp.ClientSession() as session:
        try:
            url = f"{DEXSCREENER_API_URL}?q={decoded_currency}"
            async with session.get(url, timeout=5) as response:
                data = await response.json()
                for pair in data.get('pairs', []):
                    if (pair.get('chainId') == 'xrpl' and
                        pair.get('baseToken', {}).get('symbol') == decoded_currency and
                        pair.get('quoteToken', {}).get('symbol') == 'XRP'):
                        return float(pair.get('priceNative', '0'))
                return None
        except Exception as e:
            logger.error(f"DEX Screener error for {decoded_currency}: {e}")
            return None

async def get_current_price(currency, issuer, transactions):
    """Fetch current token price in XRP with fallbacks."""
    cache_key = f"{currency}-{issuer}"
    if cache_key in price_cache:
        return price_cache[cache_key]

    decoded_currency = decode_hex_currency(currency)

    async def get_dex_price():
        try:
            buy_offers = client.request(BookOffers(
                taker_pays={"currency": "XRP"},
                taker_gets={"currency": currency, "issuer": issuer},
                limit=5  # Reduced limit for speed
            )).result.get("offers", [])
            sell_offers = client.request(BookOffers(
                taker_gets={"currency": "XRP"},
                taker_pays={"currency": currency, "issuer": issuer},
                limit=5
            )).result.get("offers", [])
            MIN_OFFERS, MIN_VOLUME = 2, 1000  # Lowered volume threshold
            buy_price = None
            if len(buy_offers) >= MIN_OFFERS:
                xrp = sum(float(o["TakerPays"]) / 1_000_000 for o in buy_offers)
                tokens = sum(float(o["TakerGets"]["value"]) for o in buy_offers)
                if tokens >= MIN_VOLUME:
                    buy_price = xrp / tokens
            sell_price = None
            if len(sell_offers) >= MIN_OFFERS:
                xrp = sum(float(o["TakerGets"]) / 1_000_000 for o in sell_offers)
                tokens = sum(float(o["TakerPays"]["value"]) for o in sell_offers)
                if tokens >= MIN_VOLUME:
                    sell_price = xrp / tokens
            if buy_price and sell_price:
                return (buy_price + sell_price) / 2
            return buy_price or sell_price or 0.000001
        except Exception as e:
            logger.error(f"XRPL DEX price error for {currency}-{issuer}: {e}")
            return 0.000001

    async def get_historical_price():
        prices = []
        cutoff = datetime.utcnow() - timedelta(days=30)
        for tx in transactions[:100]:  # Limit to 100 recent txs for speed
            tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
            if tx_time < cutoff or not tx.get('meta') or "delivered_amount" not in tx["meta"]:
                continue
            meta = tx["meta"]
            if isinstance(meta["delivered_amount"], dict) and meta["delivered_amount"]["currency"] == "XRP":
                xrp = float(meta["delivered_amount"]["value"]) / 1_000_000
                for node in meta.get("AffectedNodes", []):
                    if ("ModifiedNode" in node and
                        node["ModifiedNode"].get("LedgerEntryType") == "RippleState" and
                        node["ModifiedNode"].get("FinalFields", {}).get("Balance", {}).get("currency") == currency and
                        node["ModifiedNode"].get("HighLimit", {}).get("issuer") == issuer):
                        token = float(node["ModifiedNode"]["FinalFields"]["Balance"]["value"])
                        if token != 0:
                            prices.append(xrp / abs(token))
        return sum(prices) / len(prices) if prices else None

    tasks = [
        fetch_dexscreener_price(decoded_currency),
        asyncio.to_thread(get_dex_price),  # Run blocking XRPL calls in thread
        asyncio.to_thread(get_historical_price)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for price in results:
        if isinstance(price, float) and price > 0.000001:
            price_cache[cache_key] = price
            return price
    price = 0.000001
    price_cache[cache_key] = price
    return price

async def get_lp_token_value(issuer, amount_held, transactions):
    """Calculate LP token value based on AMM pool data."""
    try:
        cache_key = f"amm:{issuer}"
        if cache_key in amm_cache:
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
            return 0

        token_price = await get_current_price(token_currency, token_issuer, transactions)
        total_pool_value = amount_xrp + (amount_token * token_price)
        value_per_lp = total_pool_value / lp_tokens_issued
        return amount_held * value_per_lp
    except Exception as e:
        logger.error(f"Error calculating LP token value for {issuer}: {e}")
        return 0

@app.route('/token_pnl', methods=['POST'])
@limiter.limit("10 per minute")
async def get_token_pnl():
    """Calculate token PNL, separating AMM LP tokens."""
    data = request.json
    address = data.get('address')
    if not address:
        return jsonify({'error': 'Wallet address is required'}), 400

    try:
        # Check PNL cache
        cache_key = f"pnl:{address}"
        if cache_key in pnl_cache:
            return jsonify(pnl_cache[cache_key])

        # Fetch transactions (limited to 30 days)
        transactions = []
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        marker = None
        max_txs = 200  # Cap total transactions
        while len(transactions) < max_txs:
            req = AccountTx(
                account=address,
                ledger_index_min=-1,
                ledger_index_max=-1,
                limit=50,  # Reduced limit for speed
                marker=marker,
                forward=True
            )
            response = client.request(req)
            result = response.result
            for tx in result.get('transactions', []):
                tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
                if tx_time < cutoff_time:
                    break
                transactions.append(tx)
            marker = result.get('marker')
            if not marker or tx_time < cutoff_time:
                break

        # Fetch current holdings
        req = AccountLines(account=address)
        response = client.request(req)
        lines = response.result['lines']
        holdings = {f"{line['currency']}-{line['account']}": float(line['balance'])
                    for line in lines if float(line['balance']) > 0.001}

        xrp_balance = sum(float(line['balance']) for line in lines if line['currency'] == 'XRP')

        regular_tokens = []
        amm_lp_tokens = []
        for token, amount_held in holdings.items():
            currency, issuer = token.split('-')
            currency_name, is_amm_lp = decode_currency(currency, issuer)
            buys = deque()
            realized_pnl = 0.0

            for tx in transactions:
                tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
                meta = tx.get('meta', {})
                if isinstance(meta, dict):
                    changes = get_balance_changes(meta, address)
                    delta_xrp = changes.get('XRP', 0) / 1_000_000
                    if token in changes:
                        delta_token = changes[token]
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
            current_value = (await get_lp_token_value(issuer, amount_held, transactions) if is_amm_lp
                            else amount_held * await get_current_price(currency, issuer, transactions))
            unrealized_pnl = current_value - cost_basis
            total_pnl = realized_pnl + unrealized_pnl

            token_data = {
                'currency': currency_name,
                'issuer': issuer,
                'amount_held': amount_held,
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

        result = {
            'xrp_balance': xrp_balance,
            'tokens': regular_tokens,
            'amm_lp_tokens': amm_lp_tokens
        }

        # Cache result
        pnl_cache[cache_key] = result
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({
            'xrp_balance': 0,
            'tokens': [],
            'amm_lp_tokens': [],
            'error': 'Partial or no data available due to processing error'
        }), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
