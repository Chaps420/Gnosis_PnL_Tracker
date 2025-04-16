from flask import Flask, request, jsonify
from flask_cors import CORS
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountLines, AccountObjects, BookOffers, AMMInfo
from xrpl.models.requests.account_objects import AccountObjectType
from xrpl.utils import xrp_to_drops, drops_to_xrp
import logging

app = Flask(__name__)

# Enable CORS
CORS(app, resources={
    r"/token_pnl": {
        "origins": ["https://chaps420.github.io", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# XRPL client (mainnet)
XRPL_CLIENT = JsonRpcClient("https://s1.ripple.com:51234/")  # Mainnet
# XRPL_CLIENT = JsonRpcClient("https://s.altnet.rippletest.net:51234/")  # Testnet

def get_token_price_in_xrp(currency, issuer, amount=1):
    """Fetch the price of a token in XRP from the XRPL DEX order book."""
    try:
        # Define the trading pair (token/XRP)
        book_request = BookOffers(
            taker_gets={"currency": "XRP"},
            taker_pays={"currency": currency, "issuer": issuer},
            limit=10
        )
        book_response = XRPL_CLIENT.request(book_request)

        if not book_response.is_successful() or not book_response.result.get("offers"):
            logger.warning(f"No order book data for {currency}/{issuer}")
            return None

        # Calculate average price from bids (taker_pays/taker_gets)
        total_price = 0
        total_quantity = 0
        for offer in book_response.result["offers"]:
            taker_gets = float(offer["TakerGets"]["value"]) if isinstance(offer["TakerGets"], dict) else drops_to_xrp(offer["TakerGets"])
            taker_pays = float(offer["TakerPays"]["value"]) if isinstance(offer["TakerPays"], dict) else drops_to_xrp(offer["TakerPays"])
            price = taker_pays / taker_gets  # Price in XRP per token
            total_price += price * taker_gets
            total_quantity += taker_gets

        if total_quantity == 0:
            return None

        avg_price = total_price / total_quantity
        return avg_price

    except Exception as e:
        logger.error(f"Error fetching price for {currency}/{issuer}: {str(e)}")
        return None

def get_amm_lp_token_value(amm_currency, amm_issuer):
    """Calculate the value of an AMM LP token in XRP based on the pool's reserves."""
    try:
        # Fetch AMM pool info
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

        # Calculate total pool value in XRP
        pool_value_xrp = 0

        # Asset1 (XRP)
        if isinstance(asset1, str):  # XRP
            pool_value_xrp += drops_to_xrp(asset1)
        else:  # Token
            price = get_token_price_in_xrp(asset1["currency"], asset1["issuer"])
            if price:
                pool_value_xrp += float(asset1["value"]) * price

        # Asset2 (Token)
        if isinstance(asset2, str):  # XRP
            pool_value_xrp += drops_to_xrp(asset2)
        else:  # Token
            price = get_token_price_in_xrp(asset2["currency"], asset2["issuer"])
            if price:
                pool_value_xrp += float(asset2["value"]) * price

        if pool_value_xrp == 0 or lp_token_supply == 0:
            return None

        # Value per LP token in XRP
        lp_token_value = pool_value_xrp / lp_token_supply
        return lp_token_value

    except Exception as e:
        logger.error(f"Error fetching AMM LP token value for {amm_currency}/{amm_issuer}: {str(e)}")
        return None

def get_wallet_tokens(address):
    """Fetch all tokens (regular and AMM LP) held by the XRPL wallet with current value in XRP."""
    try:
        # Validate address format
        if not address.startswith("r") or len(address) < 25 or len(address) > 35:
            return {"error": "Invalid XRPL address format"}

        # Initialize response data
        response_data = {
            "tokens": [],
            "amm_lp_tokens": []
        }

        # 1. Fetch regular tokens (trust lines) using AccountLines
        account_lines_request = AccountLines(account=address)
        account_lines_response = XRPL_CLIENT.request(account_lines_request)
        
        if account_lines_response.is_successful():
            for line in account_lines_response.result.get("lines", []):
                amount_held = float(line["balance"])
                price_in_xrp = get_token_price_in_xrp(line["currency"], line["account"])
                current_value = amount_held * price_in_xrp if price_in_xrp is not None else None
                token = {
                    "currency": line["currency"],
                    "issuer": line["account"],
                    "amount_held": amount_held,
                    "current_value": round(current_value, 6) if current_value is not None else None
                }
                response_data["tokens"].append(token)
        else:
            logger.error(f"Failed to fetch account lines: {account_lines_response.result}")
            return {"error": "Failed to fetch regular tokens"}

        # 2. Fetch AMM LP tokens using AccountObjects
        account_objects_request = AccountObjects(
            account=address,
            type=AccountObjectType.AMM
        )
        account_objects_response = XRPL_CLIENT.request(account_objects_request)

        if account_objects_response.is_successful():
            for obj in account_objects_response.result.get("account_objects", []):
                if obj.get("LedgerEntryType") == "AMM":
                    currency = obj.get("LPToken", {}).get("currency", "N/A")
                    issuer = obj.get("LPToken", {}).get("issuer", "N/A")
                    amount_held = float(obj.get("LPTokenBalance", {}).get("value", 0))
                    lp_token_value = get_amm_lp_token_value(currency, issuer)
                    current_value = amount_held * lp_token_value if lp_token_value is not None else None
                    lp_token = {
                        "currency": currency,
                        "issuer": issuer,
                        "amount_held": amount_held,
                        "current_value": round(current_value, 6) if current_value is not None else None
                    }
                    response_data["amm_lp_tokens"].append(lp_token)
        else:
            logger.error(f"Failed to fetch account objects: {account_objects_response.result}")
            return {"error": "Failed to fetch AMM LP tokens"}

        return response_data

    except Exception as e:
        logger.error(f"Error fetching wallet tokens: {str(e)}")
        return {"error": f"Server error: {str(e)}"}

@app.route('/token_pnl', methods=['GET', 'POST', 'OPTIONS'])
def token_pnl():
    """API endpoint to fetch current token balances and values for a given XRPL wallet address."""
    try:
        # Handle preflight OPTIONS request
        if request.method == "OPTIONS":
            return jsonify({}), 200

        data = request.get_json()
        address = data.get("address", "").strip()

        if not address:
            return jsonify({"error": "No address provided"}), 400

        # Fetch wallet tokens with current values
        result = get_wallet_tokens(address)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in token_pnl endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
