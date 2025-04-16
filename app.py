from flask import Flask, request, jsonify
from flask_cors import CORS
import xrpl
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountTx, AccountLines
from datetime import datetime
import logging
from collections import deque
import requests
import binascii

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# XRPL client setup
JSON_RPC_URL = "https://s1.ripple.com:51234/"
client = JsonRpcClient(JSON_RPC_URL)

# Constants
START_DATE = datetime(2024, 9, 1)  # Transactions starting from September 1, 2024

def fetch_transactions_since(address, start_date):
    """Fetch transactions since a specific date."""
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
            if tx_time >= start_date:
                transactions.append(tx)
        marker = result.get('marker')
        if not marker:
            break
    return transactions

@app.route('/token_pnl', methods=['POST'])
def get_token_pnl():
    """Calculate token PNL, separating AMM LP tokens."""
    data = request.json
    address = data.get('address')

    if not address:
        return jsonify({'error': 'Wallet address is required'}), 400

    try:
        # Fetch transactions from 9/1/2024
        transactions = fetch_transactions_since(address, START_DATE)

        # Fetch account lines (current holdings)
        req = AccountLines(account=address)
        response = client.request(req)
        lines = response.result['lines']
        holdings = {f"{line['currency']}-{line['account']}": float(line['balance']) 
                    for line in lines if float(line['balance']) > 0.001}  # Filter out dust

        xrp_balance = sum(float(line['balance']) for line in lines if line['currency'] == 'XRP')

        regular_tokens = []
        amm_lp_tokens = []
        for token, amount_held in holdings.items():
            currency, issuer = token.split('-')
            buys = deque()
            realized_pnl = 0.0

            # Process transactions for token
            for tx in transactions:
                changes = get_balance_changes(tx['meta'], address)
                delta_token = changes.get(token, 0)
                if delta_token > 0:  # Buy
                    cost = 0
                    for other, delta_other in changes.items():
                        if other != token and delta_other < 0:
                            price_other = get_asset_price(other, transactions)
                            cost += -delta_other * price_other
                    price = cost / delta_token if delta_token > 0 else 0
                    buys.append({'amount': delta_token, 'price': price})
                    logger.debug(f"Buy detected for {token}: {delta_token} @ {price}")
                elif delta_token < 0:  # Sell
                    proceeds = 0
                    for other, delta_other in changes.items():
                        if other != token and delta_other > 0:
                            price_other = get_asset_price(other, transactions)
                            proceeds += delta_other * price_other
                    sell_amount = -delta_token
                    sell_value = proceeds
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
            current_value = amount_held * get_current_price(currency, issuer, transactions)
            unrealized_pnl = current_value - cost_basis if cost_basis else 0
            total_pnl = realized_pnl + unrealized_pnl

            token_data = {
                'currency': currency,
                'issuer': issuer,
                'amount_held': amount_held,
                'initial_investment': cost_basis,
                'current_value': current_value,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl
            }

            regular_tokens.append(token_data)

        return jsonify({
            'xrp_balance': xrp_balance,
            'tokens': regular_tokens
        })
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
