from flask import Flask, request, jsonify
from flask_cors import CORS
import xrpl
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountTx, AccountLines, BookOffers, AMMInfo
from datetime import datetime, timedelta
from collections import deque
import logging
import binascii
import requests

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# XRPL client setup
JSON_RPC_URL = "https://s1.ripple.com:51234/"
client = JsonRpcClient(JSON_RPC_URL)

# DEX Screener API setup
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex/search"

def decode_currency(currency, issuer):
    """Decode currency and determine if it's an AMM LP token. Treat 'Unknown' 40-hex tokens as LP tokens."""
    if len(currency) == 40 and all(c in '0123456789ABCDEFabcdef' for c in currency):
        try:
            amm_info = client.request(AMMInfo(amm_account=issuer)).result
            if "amm" in amm_info and "asset1" in amm_info["amm"] and "asset2" in amm_info["amm"]:
                asset1 = amm_info["amm"]["asset1"]["currency"]
                asset2 = amm_info["amm"]["asset2"]["currency"]
                return f"{asset1}-{asset2} LP", True
        except Exception as e:
            logger.debug(f"AMMInfo failed for {issuer}: {e}")
        
        try:
            decoded = binascii.unhexlify(currency).decode('ascii', errors='ignore').strip('\x00')
            if decoded and all(c.isprintable() for c in decoded):
                return decoded, False
            else:
                return "LP Token (Unknown Pair)", True  # Treat as LP token if decoding fails
        except Exception:
            return "LP Token (Unknown Pair)", True  # Treat as LP token on decoding exception
    return currency, False

def get_balance_changes(meta, address):
    """Extract balance changes for XRP and tokens from transaction metadata."""
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
                    delta_xrp = final_balance - previous_balance
                    changes['XRP'] += delta_xrp
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
                        if low == address:
                            changes[token_key] = changes.get(token_key, 0) + delta
                        else:
                            changes[token_key] = changes.get(token_key, 0) - delta
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
                    if low == address:
                        changes[token_key] = changes.get(token_key, 0) + balance
                    else:
                        changes[token_key] = changes.get(token_key, 0) - balance
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
                    if low == address:
                        changes[token_key] = changes.get(token_key, 0) - balance
                    else:
                        changes[token_key] = changes.get(token_key, 0) + balance
    return changes

def get_historical_price(currency, issuer, transactions):
    """Calculate average historical price from recent transactions (last 30 days)."""
    prices = []
    cutoff_time = datetime.utcnow() - timedelta(days=30)
    for tx in transactions:
        tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
        if tx_time < cutoff_time or not tx.get('meta') or "delivered_amount" not in tx["meta"]:
            continue
        meta = tx["meta"]
        delivered = meta["delivered_amount"]
        if isinstance(delivered, dict) and delivered.get("currency") == "XRP":
            xrp_amount = float(delivered["value"]) / 1_000_000
            for node in meta.get("AffectedNodes", []):
                if "ModifiedNode" in node and node["ModifiedNode"].get("LedgerEntryType") == "RippleState":
                    final_fields = node["ModifiedNode"].get("FinalFields", {})
                    if (final_fields.get("Balance", {}).get("currency") == currency and 
                        final_fields.get("HighLimit", {}).get("issuer") == issuer):
                        token_amount = float(final_fields["Balance"]["value"])
                        if token_amount != 0:
                            prices.append(xrp_amount / abs(token_amount))
    return sum(prices) / len(prices) if prices else None

def get_dexscreener_price(currency, issuer):
    """Fetch price from DEX Screener API."""
    try:
        currency_name, _ = decode_currency(currency, issuer)
        url = f"{DEXSCREENER_API_URL}?q={currency_name}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        for pair in data.get('pairs', []):
            if pair.get('chainId') == 'xrpl' and pair.get('baseToken', {}).get('symbol') == currency_name:
                pair_address = pair.get('pairAddress', '')
                parts = pair_address.split('.')
                if len(parts) > 1 and parts[1].split('_')[0].startswith(issuer[:10]):
                    if pair.get('quoteToken', {}).get('symbol') == 'XRP':
                        price = float(pair.get('priceNative', '0'))
                        logger.info(f"DEX Screener price for {currency}-{issuer}: {price} XRP")
                        return price
        return None
    except Exception as e:
        logger.error(f"Error fetching DEX Screener price for {currency}-{issuer}: {e}")
        return None

def get_dex_price(currency, issuer, transactions):
    """Fetch current price from XRPL DEX."""
    try:
        req_buy = BookOffers(
            taker_pays={"currency": "XRP"},
            taker_gets={"currency": currency, "issuer": issuer},
            limit=10
        )
        response_buy = client.request(req_buy)
        buy_offers = response_buy.result.get("offers", [])

        req_sell = BookOffers(
            taker_gets={"currency": "XRP"},
            taker_pays={"currency": currency, "issuer": issuer},
            limit=10
        )
        response_sell = client.request(req_sell)
        sell_offers = response_sell.result.get("offers", [])

        MIN_TOKEN_VOLUME = 10000
        MIN_OFFERS = 2

        buy_price = None
        if len(buy_offers) >= MIN_OFFERS:
            total_xrp_buy = sum(float(offer["TakerPays"]) / 1_000_000 for offer in buy_offers)
            total_tokens_buy = sum(float(offer["TakerGets"]["value"]) for offer in buy_offers)
            if total_tokens_buy >= MIN_TOKEN_VOLUME:
                buy_price = total_xrp_buy / total_tokens_buy

        sell_price = None
        if len(sell_offers) >= MIN_OFFERS:
            total_xrp_sell = sum(float(offer["TakerGets"]) / 1_000_000 for offer in sell_offers)
            total_tokens_sell = sum(float(offer["TakerPays"]["value"]) for offer in sell_offers)
            if total_tokens_sell >= MIN_TOKEN_VOLUME:
                sell_price = total_xrp_sell / total_tokens_sell

        if buy_price and sell_price:
            return (buy_price + sell_price) / 2
        elif buy_price:
            return buy_price
        elif sell_price:
            return sell_price
        return 0.000001
    except Exception as e:
        logger.error(f"Error fetching XRPL DEX price for {currency}-{issuer}: {e}")
        return 0.000001

def get_current_price(currency, issuer, transactions):
    """Fetch current price with fallbacks."""
    ds_price = get_dexscreener_price(currency, issuer)
    if ds_price and ds_price > 0.000001:
        return ds_price
    dex_price = get_dex_price(currency, issuer, transactions)
    if dex_price > 0.000001:
        return dex_price
    hist_price = get_historical_price(currency, issuer, transactions)
    if hist_price:
        return hist_price
    return 0.000001

@app.route('/token_pnl', methods=['POST'])
def get_token_pnl():
    """Calculate token PNL and separate AMM LP tokens, including 'Unknown' 40-hex tokens as LP tokens."""
    data = request.json
    address = data.get('address')
    days = data.get('days', 0)

    if not address:
        return jsonify({'error': 'Wallet address is required'}), 400

    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days) if days > 0 else datetime.min

        # Fetch transactions
        transactions = []
        marker = None
        while True:
            req = AccountTx(
                account=address,
                ledger_index_min=-1,
                ledger_index_max=-1,
                limit=100,
                marker=marker,
                forward=True
            )
            response = client.request(req)
            result = response.result
            for tx in result.get('transactions', []):
                tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
                transactions.append(tx)
            marker = result.get('marker')
            if not marker:
                break

        # Fetch current holdings
        req = AccountLines(account=address)
        response = client.request(req)
        lines = response.result['lines']
        holdings = {f"{line['currency']}-{line['account']}": float(line['balance']) 
                    for line in lines if float(line['balance']) > 0}

        regular_tokens = []
        amm_lp_tokens = []
        for token, amount_held in holdings.items():
            currency, issuer = token.split('-')
            currency_name, is_amm_lp = decode_currency(currency, issuer)
            buys = deque()
            realized_pnl = 0.0

            # Process transactions for buys and sells
            for tx in transactions:
                tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
                if tx_time >= start_time:
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

            # Calculate PNL metrics
            cost_basis = sum(buy['amount'] * buy['price'] for buy in buys)
            current_price = get_current_price(currency, issuer, transactions)
            current_value = amount_held * current_price
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

        # Sort by current_value descending
        regular_tokens.sort(key=lambda x: x['current_value'] if x['current_value'] is not None else 0, reverse=True)
        amm_lp_tokens.sort(key=lambda x: x['current_value'] if x['current_value'] is not None else 0, reverse=True)

        return jsonify({
            'tokens': regular_tokens,
            'amm_lp_tokens': amm_lp_tokens
        })
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
