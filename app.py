from flask import Flask, request, jsonify
from flask_cors import CORS
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountLines, BookOffers, AMMInfo, AccountTx
from xrpl.utils import xrp_to_drops, drops_to_xrp
import logging
from datetime import datetime, timezone

app = Flask(__name__)

# Enable CORS for specific origins - corrected to use dictionary for resources
CORS(app, resources={r"/token_pnl": {"origins": ["[invalid url, do not cite] "[invalid url, do not cite],
                                     "methods": ["GET", "POST", "OPTIONS"],
                                     "allow_headers": ["Content-Type", "Authorization"]}})

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# XRPL client (mainnet)
XRPL_CLIENT = JsonRpcClient("[invalid url, do not cite])  # Mainnet

# Ripple epoch for time conversion
ripple_epoch = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Start time: September 1, 2024
start_time = datetime(2024, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
start_ripple_time = int((start_time - ripple_epoch).total_seconds())  # 778464000

def get_balance_changes(meta, wallet):
    """
    Parse transaction metadata to determine balance changes for the wallet.
    
    Args:
        meta (dict): The 'meta' field from a transaction response.
        wallet (str): The XRPL wallet address.
    
    Returns:
        dict: Balance changes where keys are 'XRP' or 'currency/issuer' and values are changes.
    """
    changes = {}
    for node in meta.get('AffectedNodes', []):
        if 'ModifiedNode' in node:
            entry = node['ModifiedNode']
            if entry['LedgerEntryType'] == 'AccountRoot' and entry['FinalFields'].get('Account') == wallet:
                final_balance = int(entry['FinalFields']['Balance'])
                previous_balance = int(entry['PreviousFields'].get('Balance', final_balance))
                xrp_change = (final_balance - previous_balance) / 1_000_000
                if xrp_change != 0:
                    changes['XRP'] = changes.get('XRP', 0) + xrp_change
            elif entry['LedgerEntryType'] == 'RippleState':
                low_limit = entry['FinalFields'].get('LowLimit', {})
                high_limit = entry['FinalFields'].get('HighLimit', {})
                low_issuer = low_limit.get('issuer')
                high_issuer = high_limit.get('issuer')
                if low_issuer == wallet and high_issuer:
                    currency = entry['FinalFields']['Balance']['currency']
                    token_key = f"{currency}/{high_issuer}"
                    final_value = float(entry['FinalFields']['Balance']['value'])
                    previous_balance_dict = entry['PreviousFields'].get('Balance', {})
                    previous_value = float(previous_balance_dict.get('value', final_value))
                    token_change = -(final_value - previous_value)
                    if token_change != 0:
                        changes[token_key] = changes.get(token_key, 0) + token_change
    return changes

def get_transactions_since(address, start_ripple_time, tx_type=None):
    """
    Fetch all transactions for the wallet since the specified ripple time.
    
    Args:
        address (str): The XRPL wallet address.
        start_ripple_time (int): The start time in ripple seconds.
        tx_type (str, optional): Filter by transaction type (e.g., "OfferCreate", "AMMDeposit").
    
    Returns:
        list: List of valid transactions since the start time.
    """
    transactions = []
    marker = None
    while True:
        try:
            response = XRPL_CLIENT.request(AccountTx(
                account=address,
                limit=100,
                marker=marker
            ))
            if not response.is_successful():
                logger.error(f"Failed to fetch transactions for {address}: {response.result}")
                return transactions
            if 'transactions' not in response.result or not response.result['transactions']:
                logger.info(f"No transactions found for {address}")
                return transactions
            for tx_entry in response.result['transactions']:
                if 'tx' not in tx_entry and 'tx_json' in tx_entry:
                    tx_entry['tx'] = tx_entry['tx_json']
                if 'tx' not in tx_entry or not isinstance(tx_entry['tx'], dict):
                    logger.warning(f"Skipping transaction without valid 'tx': {tx_entry}")
                    continue
                tx = tx_entry['tx']
                if 'date' not in tx or 'TransactionType' not in tx:
                    logger.warning(f"Skipping transaction without 'date' or 'TransactionType': {tx_entry}")
                    continue
                if tx['date'] >= start_ripple_time:
                    if tx_type is None or tx['TransactionType'] == tx_type:
                        transactions.append(tx_entry)
            marker = response.result.get('marker')
            if not marker:
                break
        except Exception as e:
            logger.error(f"Error fetching transactions for {address}: {str(e)}")
            return transactions
    return transactions

def get_initial_investments_regular_tokens(address, start_ripple_time):
    """
    Calculate the total XRP spent on each regular token via OfferCreate transactions.
    
    Args:
        address (str): The XRPL wallet address.
        start_ripple_time (int): The start time in ripple seconds.
    
    Returns:
        dict: Mapping of 'currency/issuer' to total XRP spent.
    """
    initial_investments = {}
    transactions = get_transactions_since(address, start_ripple_time, tx_type="OfferCreate")
    for tx in transactions:
        changes = get_balance_changes(tx['meta'], address)
        xrp_spent = -changes.get('XRP', 0) if changes.get('XRP', 0) < 0 else 0
        if xrp_spent > 0:
            acquired_tokens = [key for key in changes if key != 'XRP' and changes[key] > 0]
            if acquired_tokens:
                for token_key in acquired_tokens:
                    initial_investments[token_key] = initial_investments.get(token_key, 0) + xrp_spent
    return initial_investments

def get_initial_investments_amm_lp_tokens(address, start_ripple_time):
    """
    Calculate the total XRP contributed to AMM pools via AMMDeposit transactions.
    
    Args:
        address (str): The XRPL wallet address.
        start_ripple_time (int): The start time in ripple seconds.
    
    Returns:
        dict: Mapping of 'lp_currency/lp_issuer' to total XRP contributed.
    """
    initial_investments = {}
    transactions = get_transactions_since(address, start_ripple_time, tx_type="AMMDeposit")
    for tx in transactions:
        if tx['meta']['TransactionResult'] == "tesSUCCESS":
            lp_token_key = None
            xrp_contributed = 0
            token_contributed_value = 0
            # Find LP token received
            for node in tx['meta']['AffectedNodes']:
                if 'CreatedNode' in node and node['CreatedNode']['LedgerEntryType'] == 'AMM':
                    lp_token_key = f"{node['CreatedNode']['NewFields']['LPToken']['currency']}/{node['CreatedNode']['NewFields']['LPToken']['issuer']}"
                elif 'ModifiedNode' in node and node['ModifiedNode']['LedgerEntryType'] == 'AMM':
                    lp_token_key = f"{node['ModifiedNode']['FinalFields']['LPToken']['currency']}/{node['ModifiedNode']['FinalFields']['LPToken']['issuer']}"
            # Calculate assets contributed
            changes = get_balance_changes(tx['meta'], address)
            for key, change in changes.items():
                if key == 'XRP' and change < 0:
                    xrp_contributed = -change
                elif '/' in key and change < 0:
                    currency, issuer = key.split('/')
                    price = get_token_price_in_xrp(currency, issuer)
                    if price:
                        token_contributed_value += (-change) * price
                    else:
                        logger.warning(f"No price data for {currency}/{issuer}, marking investment as None")
                        token_contributed_value = None
                        break
            total_contributed_xrp = None
            if xrp_contributed > 0 or token_contributed_value is not None:
                total_contributed_xrp = xrp_contributed + (token_contributed_value if token_contributed_value is not None else 0)
            if lp_token_key and total_contributed_xrp is not None:
                initial_investments[lp_token_key] = initial_investments.get(lp_token_key, 0) + total_contributed_xrp
            elif lp_token_key:
                initial_investments[lp_token_key] = None
    return initial_investments

def get_token_price_in_xrp(currency, issuer):
    """
    Get the current price of a token in XRP based on order book data.
    
    Args:
        currency (str): Token currency code.
        issuer (str): Token issuer address.
    
    Returns:
        float or None: Average price in XRP per token, or None if unavailable.
    """
    try:
        book_request = BookOffers(
            taker_gets={"currency": "XRP"},
            taker_pays={"currency": currency, "issuer": issuer},
            limit=10
        )
        book_response = XRPL_CLIENT.request(book_request)
        if not book_response.is_successful() or not book_response.result.get("offers"):
            logger.warning(f"No order book data for {currency}/{issuer}")
            return None
        total_price = 0
        total_quantity = 0
        for offer in book_response.result["offers"]:
            taker_gets = float(drops_to_xrp(offer["TakerGets"]))
            taker_pays = float(offer["TakerPays"]["value"])
            price = taker_gets / taker_pays  # XRP per token
            total_price += price * taker_pays
            total_quantity += taker_pays
        if total_quantity == 0:
            return None
        return total_price / total_quantity
    except Exception as e:
        logger.error(f"Error fetching price for {currency}/{issuer}: {str(e)}")
        return None

def get_amm_lp_token_value(amm_currency, amm_issuer):
    """
    Calculate the value of an AMM LP token in XRP.
    
    Args:
        amm_currency (str): LP token currency code.
        amm_issuer (str): LP token issuer address.
    
    Returns:
        float or None: Value per LP token in XRP, or None if unavailable.
    """
    try:
        amm_request = AMMInfo(
            asset={"currency": "XRP"},
            asset2={"currency": amm_currency, "issuer": amm_issuer}
        )
        amm_response = XRPL_CLIENT.request(amm_request)
        if not amm_response.is_successful() or not amm_response.result.get("amm"):
            logger.warning(f"No AMM pool data for {amm_currency}/{amm_issuer}")
            return None
        amm_data = amm_response.result["amm"]
        lp_token_supply = float(amm_data["lp_token"]["value"])
        asset1 = amm_data["amount"]
        asset2 = amm_data["amount2"]
        pool_value_xrp = 0
        if isinstance(asset1, str):  # XRP
            pool_value_xrp += float(drops_to_xrp(asset1))
        else:  # Token
            price = get_token_price_in_xrp(asset1["currency"], asset1["issuer"])
            if price:
                pool_value_xrp += float(asset1["value"]) * price
        if isinstance(asset2, str):  # XRP
            pool_value_xrp += float(drops_to_xrp(asset2))
        else:  # Token
            price = get_token_price_in_xrp(asset2["currency"], asset2["issuer"])
            if price:
                pool_value_xrp += float(asset2["value"]) * price
        if pool_value_xrp == 0 or lp_token_supply == 0:
            return None
        return pool_value_xrp / lp_token_supply
    except Exception as e:
        logger.error(f"Error fetching AMM LP token value for {amm_currency}/{amm_issuer}: {str(e)}")
        return None

def get_wallet_tokens(address):
    """
    Retrieve token balances and initial investments for the wallet.
    
    Args:
        address (str): The XRPL wallet address.
    
    Returns:
        dict: Token data including balances and initial investments.
    """
    try:
        if not address.startswith("r") or len(address) < 25 or len(address) > 35:
            return {"error": "Invalid XRPL address format"}

        response_data = {"tokens": [], "amm_lp_tokens": []}

        # Fetch all tokens using AccountLines
        account_lines_request = AccountLines(account=address)
        account_lines_response = XRPL_CLIENT.request(account_lines_request)
        if account_lines_response.is_successful():
            for line in account_lines_response.result.get("lines", []):
                if "account" not in line:
                    logger.warning(f"Missing 'account' in line: {line}")
                    continue
                amount_held = float(line["balance"])
                price_in_xrp = get_token_price_in_xrp(line["currency"], line["account"])
                current_value = amount_held * price_in_xrp if price_in_xrp is not None else None
                token = {
                    "currency": line["currency"],
                    "issuer": line["account"],
                    "amount_held": amount_held,
                    "current_value": round(current_value, 6) if current_value is not None else None,
                    "initial_investment": None
                }
                if len(line["currency"]) == 40:
                    response_data["amm_lp_tokens"].append(token)
                else:
                    response_data["tokens"].append(token)
        else:
            logger.error(f"Failed to fetch account lines: {account_lines_response.result}")
            return {"error": "Failed to fetch token balances"}

        # Calculate initial investments for regular tokens
        initial_investments_regular = get_initial_investments_regular_tokens(address, start_ripple_time)
        for token in response_data["tokens"]:
            token_key = f"{token['currency']}/{token['issuer']}"
            token['initial_investment'] = initial_investments_regular.get(token_key, 0)

        # Calculate initial investments for AMM LP tokens
        initial_investments_amm = get_initial_investments_amm_lp_tokens(address, start_ripple_time)
        for lp_token in response_data["amm_lp_tokens"]:
            lp_token_key = f"{lp_token['currency']}/{lp_token['issuer']}"
            lp_token['initial_investment'] = initial_investments_amm.get(lp_token_key, None)

        return response_data

    except Exception as e:
        logger.error(f"Error fetching wallet tokens: {str(e)}")
        return {"error": f"Server error: {str(e)}"}

@app.route('/token_pnl', methods=['POST', 'OPTIONS'])
def token_pnl():
    """
    Flask endpoint to handle token profit/loss requests.
    
    Returns:
        JSON response with token data or error message.
    """
    try:
        if request.method == "OPTIONS":
            return jsonify({}), 200

        data = request.get_json()
        address = data.get("address", "").strip()

        if not address:
            return jsonify({"error": "No address provided"}), 400

        result = get_wallet_tokens(address)
        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in token_pnl endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Ensure CORS headers for all responses
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '[invalid url, do not cite]
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
