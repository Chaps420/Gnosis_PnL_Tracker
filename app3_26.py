from flask import Flask, request, jsonify
from flask_cors import CORS
import xrpl
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountTx, AccountLines, BookOffers, AMMInfo
from datetime import datetime, timedelta
from collections import deque
import logging
import binascii

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# XRPL client setup
JSON_RPC_URL = "https://s1.ripple.com:51234/"
client = JsonRpcClient(JSON_RPC_URL)

# Helper Functions
def decode_currency(currency, issuer):
    """Decode currency and label tokens appropriately."""
    if len(currency) == 40 and all(c in '0123456789ABCDEF' for c in currency.upper()):
        try:
            amm_info = client.request(AMMInfo(amm_account=issuer)).result
            if "amm" in amm_info and "asset1" in amm_info["amm"] and "asset2" in amm_info["amm"]:
                asset1 = amm_info["amm"]["asset1"]["currency"]
                asset2 = amm_info["amm"]["asset2"]["currency"]
                return f"{asset1}-{asset2} LP"
        except Exception as e:
            logger.debug(f"AMMInfo failed for {issuer}: {e}")
        
        try:
            decoded = bytes.fromhex(currency).decode('ascii', errors='ignore').strip('\x00')
            if decoded and all(c.isprintable() for c in decoded):
                return decoded
            else:
                return "Unknown Token"
        except Exception:
            return "Unknown Token"
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
                        if "delivered_amount" in tx["meta"]:
                            delivered = tx["meta"]["delivered_amount"]
                            if isinstance(delivered, dict) and delivered.get("currency") == "XRP":
                                xrp_amount = float(delivered["value"]) / 1_000_000
                                token_amount = float(final_fields["Balance"]["value"])
                                if token_amount != 0:
                                    prices.append(xrp_amount / abs(token_amount))
    return sum(prices) / len(prices) if prices else None

def get_lp_token_value(issuer, amount_held):
    """Calculate LP token value in XRP based on AMM pool data."""
    try:
        amm_info = client.request(AMMInfo(amm_account=issuer)).result
        if "amm" not in amm_info:
            logger.error(f"No AMM data found for {issuer}")
            return 0
        amm_data = amm_info["amm"]
        if "asset1" not in amm_data or "asset2" not in amm_data:
            logger.error(f"Missing asset1 or asset2 in AMM data for {issuer}")
            return 0
        asset1 = amm_data["asset1"]
        asset2 = amm_data["asset2"]
        lp_token_supply = float(amm_data["lp_token"]["value"])

        # Calculate value of asset1 in XRP
        if asset1["currency"] == "XRP":
            asset1_value = float(asset1["value"]) / 1_000_000
        else:
            asset1_price = get_current_price(asset1["currency"], asset1["issuer"], [])
            asset1_value = float(asset1["value"]) * asset1_price if asset1_price else 0

        # Calculate value of asset2 in XRP
        if asset2["currency"] == "XRP":
            asset2_value = float(asset2["value"]) / 1_000_000
        else:
            asset2_price = get_current_price(asset2["currency"], asset2["issuer"], [])
            asset2_value = float(asset2["value"]) * asset2_price if asset2_price else 0

        total_pool_value = asset1_value + asset2_value
        value_per_lp_token = total_pool_value / lp_token_supply if lp_token_supply != 0 else 0
        current_value = amount_held * value_per_lp_token
        logger.info(f"LP token value for {issuer}: {current_value} XRP")
        return current_value
    except Exception as e:
        logger.error(f"Error calculating LP token value for {issuer}: {e}")
        return 0

def get_current_price(currency, issuer, transactions):
    """Fetch current price from DEX with historical fallback, or LP token value if applicable."""
    # Check if it's an LP token
    if len(currency) == 40 and all(c in '0123456789ABCDEF' for c in currency.upper()):
        try:
            amm_info = client.request(AMMInfo(amm_account=issuer)).result
            if "amm" in amm_info:
                return get_lp_token_value(issuer, 1)  # Price per token
        except Exception:
            pass  # Not an LP token or AMMInfo failed, proceed to standard pricing

    try:
        # Buy offers (pay XRP, get token)
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

        # Default price if no data
        logger.warning(f"No price data available for {currency}-{issuer}, using default price")
        return 0.000001  # Small default price to avoid zero
    except Exception as e:
        logger.error(f"Error fetching DEX price for {currency}-{issuer}: {e}")
        return 0.000001

# API Endpoint
@app.route('/token_pnl', methods=['POST'])
def get_token_pnl():
    """Calculate PNL for tokens in the specified wallet and sort by current value in descending order."""
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
            currency_name = decode_currency(currency, issuer)
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
            current_value = amount_held * current_price if current_price else 0
            unrealized_pnl = current_value - cost_basis if current_price else None
            total_pnl = realized_pnl + unrealized_pnl if unrealized_pnl is not None else None

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

        # Sort token_pnl by current_value in descending order
        token_pnl.sort(key=lambda x: x['current_value'] if x['current_value'] is not None else 0, reverse=True)

        return jsonify({'tokens': token_pnl})
    except xrpl.clients.XrplException as e:
        logger.error(f"XRPL error: {e}")
        return jsonify({'error': f"XRPL error: {str(e)}"}), 503
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
