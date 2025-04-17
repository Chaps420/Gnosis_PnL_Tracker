from flask import Flask, request, jsonify
from flask_cors import CORS
import xrpl
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountTx, AccountLines, BookOffers, AMMInfo
from datetime import datetime, timedelta
from collections import deque
import logging
import requests
import binascii
import time
from cachetools import TTLCache

app = Flask(__name__)
CORS(app)  # Simple CORS setup to allow all origins

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# XRPL client setup
JSON_RPC_URL = "https://s1.ripple.com:51234/"
client = JsonRpcClient(JSON_RPC_URL)

# DEX Screener API setup
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex/search"

# Required tokens for access
REQUIRED_TOKENS = {"UGA", "GNOSIS", "CULT", "OBEY", "FPT", "METH"}

# Initialize caches
account_tx_cache = TTLCache(maxsize=100, ttl=300)  # Transactions: 5 minutes
account_lines_cache = TTLCache(maxsize=100, ttl=300)  # Balances: 5 minutes
amm_info_cache = TTLCache(maxsize=1000, ttl=1800)  #
price_cache = TTLCache(maxsize=1000, ttl=60)  # Prices: 1 minute
pnl_cache = TTLCache(maxsize=100, ttl=300)  # PNL responses: 5 minutes

def decode_hex_currency(hex_code):
    """Decode a 40-character hex currency code to ASCII."""
    if len(hex_code) == 40 and all(c in '0123456789ABCDEFabcdef' for c in hex_code):
        try:
            bytes_code = binascii.unhexlify(hex_code)
            return bytes_code.split(b'\0', 1)[0].decode('ascii')
        except Exception:
            pass
    return hex_code

def decode_currency(currency, issuer):
    """Decode currency and identify AMM LP tokens using AMMInfo with caching."""
    if len(currency) != 40 or not all(c in '0123456789ABCDEFabcdef' for c in currency):
        return currency, False

    cache_key = f"amm_info:{issuer}"
    if cache_key in amm_info_cache:
        logger.debug(f"Cache hit for AMMInfo: {issuer}")
        amm_info = amm_info_cache[cache_key]
        if amm_info is None:
            return decode_hex_currency(currency), False
    else:
        try:
            response = client.request(AMMInfo(amm_account=issuer))
            if not response.is_successful():
                logger.debug(f"AMMInfo request failed for {issuer}: {response.result}")
                amm_info_cache[cache_key] = None
                return decode_hex_currency(currency), False
            amm_info = response.result.get("amm")
            if not amm_info:
                logger.debug(f"No 'amm' field in AMMInfo response for {issuer}")
                amm_info_cache[cache_key] = None
                return decode_hex_currency(currency), False
            amm_info_cache[cache_key] = amm_info
        except Exception as e:
            logger.debug(f"AMMInfo failed for {issuer}: {e}")
            amm_info_cache[cache_key] = None
            return decode_hex_currency(currency), False

    required_fields = ["lp_token", "amount", "amount2"]
    if all(field in amm_info for field in required_fields):
        asset1 = amm_info["amount"]
        asset2 = amm_info["amount2"]
        asset1_str = "XRP" if isinstance(asset1, str) else decode_hex_currency(asset1["currency"])
        asset2_str = "XRP" if isinstance(asset2, str) else decode_hex_currency(asset2["currency"])
        return f"LP_{asset1_str}_{asset2_str}", True
    else:
        logger.debug(f"Missing required fields in AMMInfo for {issuer}: {amm_info}")
        return decode_hex_currency(currency), False

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
                    balance = float(final_fields.get('Balance', {}).get('value', 0))
                    changes[token_key] = changes.get(token_key, 0) + (-balance if low == address else balance)
    return changes

def get_current_price(currency, issuer, transactions):
    """Fetch current token price in XRP with fallbacks and caching."""
    cache_key = f"price:{currency}:{issuer}"
    if cache_key in price_cache:
        logger.debug(f"Cache hit for price: {currency}-{issuer}")
        return price_cache[cache_key]

    decoded_currency = decode_hex_currency(currency)

    def get_dexscreener_price():
        try:
            url = f"{DEXSCREENER_API_URL}?q={decoded_currency}"
            response = requests.get(url, timeout=5).json()
            for pair in response.get('pairs', []):
                if (pair.get('chainId') == 'xrpl' and 
                    pair.get('baseToken', {}).get('symbol') == decoded_currency and 
                    pair.get('quoteToken', {}).get('symbol') == 'XRP'):
                    return float(pair.get('priceNative', '0'))
            return None
        except Exception as e:
            logger.error(f"DEX Screener error for {decoded_currency}-{issuer}: {e}")
            return None

    def get_dex_price():
        try:
            buy_offers = client.request(BookOffers(
                taker_pays={"currency": "XRP"},
                taker_gets={"currency": currency, "issuer": issuer},
                limit=10
            )).result.get("offers", [])
            sell_offers = client.request(BookOffers(
                taker_gets={"currency": "XRP"},
                taker_pays={"currency": currency, "issuer": issuer},
                limit=10
            )).result.get("offers", [])

            MIN_OFFERS, MIN_VOLUME = 2, 10000
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

    def get_historical_price():
        prices = []
        cutoff = datetime.utcnow() - timedelta(days=30)
        for tx in transactions:
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

    for method in (get_dexscreener_price, get_dex_price, get_historical_price):
        price = method()
        if price and price > 0.000001:
            price_cache[cache_key] = price
            return price
    price_cache[cache_key] = 0.000001
    return 0.000001

def get_lp_token_value(issuer, amount_held, transactions):
    """Calculate LP token value based on AMM pool data."""
    try:
        cache_key = f"amm_info:{issuer}"
        if cache_key in amm_info_cache:
            logger.debug(f"Cache hit for AMMInfo in get_lp_token_value: {issuer}")
            amm_info = amm_info_cache[cache_key]
            if amm_info is None:
                return 0
        else:
            response = client.request(AMMInfo(amm_account=issuer))
            if not response.is_successful():
                logger.debug(f"AMMInfo request failed for {issuer}: {response.result}")
                amm_info_cache[cache_key] = None
                return 0
            amm_info = response.result.get("amm")
            if not amm_info:
                logger.debug(f"No 'amm' field in AMMInfo response for {issuer}")
                amm_info_cache[cache_key] = None
                return 0
            amm_info_cache[cache_key] = amm_info

        if "lp_token" not in amm_info or "amount" not in amm_info or "amount2" not in amm_info:
            logger.debug(f"Missing required fields in AMMInfo for {issuer}: {amm_info}")
            return 0
        lp_tokens_issued = float(amm_info["lp_token"]["value"])
        asset1 = amm_info["amount"]
        asset2 = amm_info["amount2"]

        if isinstance(asset1, str):  # XRP is asset1
            amount_xrp = float(asset1) / 1_000_000
            token_currency = asset2["currency"]
            token_issuer = asset2["issuer"]
            amount_token = float(asset2["value"])
        elif isinstance(asset2, str):  # XRP is asset2
            amount_xrp = float(asset2) / 1_000_000
            token_currency = asset1["currency"]
            token_issuer = asset1["issuer"]
            amount_token = float(asset1["value"])
        else:
            logger.warning(f"Both assets are tokens for {issuer}, not supported")
            return 0

        token_price = get_current_price(token_currency, token_issuer, transactions)
        total_pool_value = amount_xrp + (amount_token * token_price)
        value_per_lp = total_pool_value / lp_tokens_issued if lp_tokens_issued > 0 else 0
        return amount_held * value_per_lp
    except Exception as e:
        logger.error(f"Error calculating LP token value for {issuer}: {e}")
        return 0

@app.route('/token_pnl', methods=['POST'])
def get_token_pnl():
    """Calculate token PNL, separating AMM LP tokens, with token gating and caching."""
    data = request.json
    address = data.get('address')

    if not address:
        return jsonify({'error': 'Wallet address is required'}), 400

    # Check PNL cache
    cache_key = f"pnl:{address}"
    if cache_key in pnl_cache:
        logger.debug(f"Cache hit for PNL: {address}")
        return jsonify(pnl_cache[cache_key])

    try:
        # Fetch current holdings with retry
        cache_key_lines = f"account_lines:{address}"
        if cache_key_lines in account_lines_cache:
            logger.debug(f"Cache hit for AccountLines: {address}")
            lines = account_lines_cache[cache_key_lines]
        else:
            req = AccountLines(account=address)
            max_retries = 3
            retry_delay = 2  # seconds
            for attempt in range(max_retries):
                try:
                    response = client.request(req)
                    if not hasattr(response, 'result'):
                        logger.error(f"XRPL response missing 'result' for AccountLines: {response.__dict__}")
                        raise ValueError("Response missing 'result' key")
                    result = response.result
                    if 'error' in result:
                        logger.error(f"XRPL error for AccountLines: {result['error']}")
                        raise ValueError(f"XRPL error: {result['error']}")
                    lines = result.get('lines', [])
                    break  # Success, exit retry loop
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for AccountLines: {str(e)}")
                    if attempt + 1 == max_retries:
                        logger.error(f"Failed to fetch AccountLines after {max_retries} attempts: {str(e)}")
                        return jsonify({'error': 'Failed to fetch account lines from XRPL'}), 500
                    time.sleep(retry_delay)
            account_lines_cache[cache_key_lines] = lines

        # Check for required tokens
        has_required_token = False
        for line in lines:
            if line.get('currency', '').upper() in REQUIRED_TOKENS:
                has_required_token = True
                break
        if not has_required_token:
            logger.info(f"Access denied for {address}: No required tokens (UGA, GNOSIS, CULT, OBEY, FPT, METH)")
            return jsonify({
                'error': 'Wallet must hold UGA, GNOSIS, CULT, OBEY, FPT, or METH to access PNL'
            }), 403

        # Fetch transactions with retry
        transactions = []
        cache_key_tx = f"account_tx:{address}"
        if cache_key_tx in account_tx_cache:
            logger.debug(f"Cache hit for AccountTx: {address}")
            transactions = account_tx_cache[cache_key_tx]
        else:
            marker = None
            for attempt in range(max_retries):
                try:
                    req = AccountTx(
                        account=address,
                        ledger_index_min=-1,
                        ledger_index_max=-1,
                        limit=100,
                        marker=marker,
                        forward=True
                    )
                    response = client.request(req)
                    if not hasattr(response, 'result'):
                        logger.error(f"XRPL response missing 'result' for AccountTx: {response.__dict__}")
                        raise ValueError("Response missing 'result' key")
                    result = response.result
                    if 'error' in result:
                        logger.error(f"XRPL error for AccountTx: {result['error']}")
                        raise ValueError(f"XRPL error: {result['error']}")
                    for tx in result.get('transactions', []):
                        tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
                        transactions.append(tx)
                    marker = result.get('marker')
                    break  # Success, exit retry loop
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for AccountTx: {str(e)}")
                    if attempt + 1 == max_retries:
                        logger.error(f"Failed to fetch AccountTx after {max_retries} attempts: {str(e)}")
                        return jsonify({'error': 'Failed to fetch transactions from XRPL'}), 500
                    time.sleep(retry_delay)
            account_tx_cache[cache_key_tx] = transactions

        holdings = {f"{line['currency']}-{line['account']}": float(line['balance']) 
                    for line in lines if float(line['balance']) > 0.001}  # Filter dust

        xrp_balance = sum(float(line['balance']) for line in lines if line['currency'] == 'XRP')

        regular_tokens = []
        amm_lp_tokens = []
        for token, amount_held in holdings.items():
            currency, issuer = token.split('-')
            currency_name, is_amm_lp = decode_currency(currency, issuer)
            buys = deque()
            realized_pnl = 0.0

            # Process transactions
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
            current_value = (get_lp_token_value(issuer, amount_held, transactions) if is_amm_lp 
                            else amount_held * get_current_price(currency, issuer, transactions))
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

        # Sort by current_value
        regular_tokens.sort(key=lambda x: x['current_value'] if x['current_value'] is not None else 0, reverse=True)
        amm_lp_tokens.sort(key=lambda x: x['current_value'] if x['current_value'] is not None else 0, reverse=True)

        response = {
            'xrp_balance': xrp_balance,
            'tokens': regular_tokens,
            'amm_lp_tokens': amm_lp_tokens
        }
        pnl_cache[cache_key] = response
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
