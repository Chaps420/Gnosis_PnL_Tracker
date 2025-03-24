from flask import Flask, request, jsonify
from flask_cors import CORS
import xrpl
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountTx, AccountLines, BookOffers
from datetime import datetime, timedelta
from collections import deque
import logging
import binascii

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# XRPL client setup
JSON_RPC_URL = "https://s1.ripple.com:51234/"
client = JsonRpcClient(JSON_RPC_URL)

def decode_currency(currency):
    """Decode a 40-character hex currency code to ASCII, if applicable."""
    try:
        if len(currency) == 40 and all(c in '0123456789ABCDEF' for c in currency):
            decoded = binascii.unhexlify(currency).decode('ascii').strip('\x00')
            return decoded if decoded else currency
        return currency
    except Exception:
        return currency

def get_balance_changes(meta, address):
    """Extract balance changes for XRP and tokens from transaction metadata."""
    changes = {'XRP': 0}
    for node in meta.get('AffectedNodes', []):
        if 'ModifiedNode' in node:
            modified = node['ModifiedNode']
            if modified.get('LedgerEntryType') == 'AccountRoot':
                final_fields = modified.get('FinalFields', {})
                previous_fields = modified.get('PreviousFields', {})
                if final_fields.get('Account') == address and 'Balance' in previous_fields:
                    final_balance = int(final_fields.get('Balance', 0))
                    previous_balance = int(previous_fields['Balance'])
                    changes['XRP'] += final_balance - previous_balance
            elif modified.get('LedgerEntryType') == 'RippleState':
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
    logger.debug(f"Balance changes for address {address}: {changes}")
    return changes

def get_historical_price(currency, issuer, transactions):
    """Calculate average historical price from wallet transactions."""
    prices = []
    for tx in transactions:
        if tx.get("meta") and "AffectedNodes" in tx["meta"]:
            for node in tx["meta"]["AffectedNodes"]:
                if "ModifiedNode" in node and node["ModifiedNode"].get("LedgerEntryType") == "RippleState":
                    final_fields = node["ModifiedNode"].get("FinalFields", {})
                    if (final_fields.get("Balance", {}).get("currency") == currency and 
                        (final_fields.get("HighLimit", {}).get("issuer") == issuer or 
                         final_fields.get("LowLimit", {}).get("issuer") == issuer)):
                        # Simplified: Assuming transaction involves XRP and token
                        if "delivered_amount" in tx["meta"]:
                            delivered = tx["meta"]["delivered_amount"]
                            if isinstance(delivered, dict) and delivered.get("currency") == "XRP":
                                xrp_amount = float(delivered["value"]) / 1_000_000
                                token_amount = float(final_fields["Balance"]["value"])
                                if token_amount != 0:
                                    prices.append(xrp_amount / abs(token_amount))
    return sum(prices) / len(prices) if prices else None

def get_current_price(currency, issuer, transactions):
    """Fetch current price from DEX with historical fallback."""
    try:
        # Try DEX first: Buy offers (pay XRP, get token)
        req_buy = BookOffers(
            taker_pays={"currency": "XRP"},
            taker_gets={"currency": currency, "issuer": issuer},
            limit=1
        )
        response_buy = client.request(req_buy)
        buy_offers = response_buy.result.get("offers", [])

        # Sell offers (pay token, get XRP)
        req_sell = BookOffers(
            taker_gets={"currency": "XRP"},
            taker_pays={"currency": currency, "issuer": issuer},
            limit=1
        )
        response_sell = client.request(req_sell)
        sell_offers = response_sell.result.get("offers", [])

        buy_price = None
        sell_price = None

        if buy_offers:
            best_buy = buy_offers[0]
            xrp_amount = float(best_buy["TakerPays"]) / 1_000_000  # XRP in drops
            token_amount = float(best_buy["TakerGets"]["value"])
            buy_price = xrp_amount / token_amount if token_amount != 0 else None

        if sell_offers:
            best_sell = sell_offers[0]
            xrp_amount = float(best_sell["TakerGets"]) / 1_000_000  # XRP in drops
            token_amount = float(best_sell["TakerPays"]["value"])
            sell_price = xrp_amount / token_amount if token_amount != 0 else None

        dex_price = (buy_price + sell_price) / 2 if buy_price and sell_price else buy_price or sell_price
        if dex_price:
            logger.info(f"Using DEX price for {currency}-{issuer}: {dex_price}")
            return dex_price

        # Fallback to historical price
        hist_price = get_historical_price(currency, issuer, transactions)
        if hist_price:
            logger.info(f"Using historical price for {currency}-{issuer}: {hist_price}")
            return hist_price
        logger.warning(f"No price data available for {currency}-{issuer}")
        return None

    except Exception as e:
        logger.error(f"Error fetching DEX price for {currency}-{issuer}: {e}")
        # Fallback to historical price on error
        hist_price = get_historical_price(currency, issuer, transactions)
        if hist_price:
            logger.info(f"Using historical price (error fallback) for {currency}-{issuer}: {hist_price}")
            return hist_price
        return None

@app.route('/token_pnl', methods=['POST'])
def get_token_pnl():
    """Calculate PNL for tokens in the specified wallet."""
    data = request.json
    address = data.get('address')
    days = data.get('days', 0)

    if not address:
        return jsonify({'error': 'Wallet address is required'}), 400

    try:
        # Define time range
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
                tx_data = tx.get('tx', {})
                tx_time = datetime.utcfromtimestamp(tx_data.get('date', 0) + 946684800)
                transactions.append(tx)
            marker = result.get('marker')
            if not marker:
                break

        logger.info(f"Total transactions fetched: {len(transactions)}")

        # Get current token holdings
        req = AccountLines(account=address)
        response = client.request(req)
        lines = response.result['lines']
        holdings = {}
        for line in lines:
            currency = line['currency']
            issuer = line['account']
            amount = float(line['balance'])
            if amount > 0:
                holdings[f"{currency}-{issuer}"] = amount

        # Calculate PNL for each token
        token_pnl = []
        for token, amount_held in holdings.items():
            currency, issuer = token.split('-')
            currency_name = decode_currency(currency)
            buys = deque()
            realized_pnl = 0.0
            starting_balance = 0.0

            # Process transactions
            for tx in transactions:
                tx_time = datetime.utcfromtimestamp(tx.get('tx', {}).get('date', 0) + 946684800)
                if tx_time < start_time:
                    meta = tx.get('meta', {})
                    if isinstance(meta, dict):
                        changes = get_balance_changes(meta, address)
                        if token in changes:
                            starting_balance += changes[token]
                elif tx_time >= start_time:
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

            # Calculate cost basis and PNL
            cost_basis = sum(buy['amount'] * buy['price'] for buy in buys)
            current_price = get_current_price(currency, issuer, transactions)
            if current_price is not None:
                current_value = amount_held * current_price
                unrealized_pnl = current_value - cost_basis
                total_pnl = realized_pnl + unrealized_pnl
            else:
                current_value = None
                unrealized_pnl = None
                total_pnl = None

            token_pnl.append({
                'currency': currency_name,
                'issuer': issuer,
                'starting_balance': starting_balance,
                'amount_held': amount_held,
                'initial_investment': cost_basis,
                'current_value': current_value,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl
            })

        return jsonify({'tokens': token_pnl})
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
