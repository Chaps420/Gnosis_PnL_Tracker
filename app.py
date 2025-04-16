from flask import Flask, request, jsonify
from flask_cors import CORS
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountLines, AccountObjects, BookOffers, AMMInfo
from xrpl.models.requests.account_objects import AccountObjectType
from xrpl.utils import xrp_to_drops, drops_to_xrp
import logging

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for specific origins
CORS(app, resources={
    r"/token_pnl": {
        "origins": ["https://chaps420.github.io", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# XRPL client (mainnet)
XRPL_CLIENT = JsonRpcClient("https://s1.ripple.com:51234/")  # Mainnet

def get_token_price_in_xrp(currency, issuer):
    """Fetch the price of a token in XRP from the XRPL DEX order book using mid-price."""
    try:
        # Fetch bid order book (buying token with XRP)
        book_request_bids = BookOffers(
            taker_gets={"currency": "XRP"},
            taker_pays={"currency": currency, "issuer": issuer},
            limit=1
        )
        book_response_bids = XRPL_CLIENT.request(book_request_bids)
        bid_price = None
        if book_response_bids.is_successful() and book_response_bids.result.get("offers"):
            offer = book_response_bids.result["offers"][0]
            taker_gets_xrp = float(drops_to_xrp(offer["TakerGets"]))  # XRP amount
            taker_pays_token = float(offer["TakerPays"]["value"])  # Token amount
            if taker_pays_token != 0:
                bid_price = taker_gets_xrp / taker_pays_token  # XRP per token
                logger.info(f"Bid price for {currency}/{issuer}: {bid_price} XRP/token "
                           f"(XRP: {taker_gets_xrp}, Token: {taker_pays_token})")

        # Fetch ask order book (selling token for XRP)
        book_request_asks = BookOffers(
            taker_gets={"currency": currency, "issuer": issuer},
            taker_pays={"currency": "XRP"},
            limit=1
        )
        book_response_asks = XRPL_CLIENT.request(book_request_asks)
        ask_price = None
        if book_response_asks.is_successful() and book_response_asks.result.get("offers"):
            offer = book_response_asks.result["offers"][0]
            taker_gets_token = float(offer["TakerGets"]["value"])  # Token amount
            taker_pays_xrp = float(drops_to_xrp(offer["TakerPays"]))  # XRP amount
            if taker_gets_token != 0:
                ask_price = taker_pays_xrp / taker_gets_token  # XRP per token
                logger.info(f"Ask price for {currency}/{issuer}: {ask_price} XRP/token "
                           f"(Token: {taker_gets_token}, XRP: {taker_pays_xrp})")

        # Calculate mid-price or fallback
        if bid_price is not None and ask_price is not None:
            mid_price = (bid_price + ask_price) / 2
            logger.info(f"Mid price for {currency}/{issuer}: {mid_price} XRP/token")
            return mid_price
        elif bid_price is not None:
            logger.info(f"Using bid price for {currency}/{issuer}: {bid_price} XRP/token")
            return bid_price
        elif ask_price is not None:
            logger.info(f"Using ask price for {currency}/{issuer}: {ask_price} XRP/token")
            return ask_price
        else:
            logger.warning(f"No valid order book data for {currency}/{issuer}")
            return None

    except Exception as e:
        logger.error(f"Error fetching price for {currency}/{issuer}: {str(e)}")
        return None

def get_amm_lp_token_value(amm_currency, amm_issuer):
    """Calculate the value of an AMM LP token in XRP based on pool reserves."""
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
            logger.warning(f"Invalid pool value or LP supply for {amm_currency}/{amm_issuer}")
            return None

        lp_token_value = pool_value_xrp / lp_token_supply
        logger.info(f"LP token value for {amm_currency}/{amm_issuer}: {lp_token_value} XRP")
        return lp_token_value

    except Exception as e:
        logger.error(f"Error fetching AMM LP value for {amm_currency}/{amm_issuer}: {str(e)}")
        return None

def get_wallet_tokens(address):
    """Fetch all tokens held by the XRPL wallet with current value in XRP."""
    try:
        if not address.startswith("r") or len(address) < 25 or len(address) > 35:
            return {"error": "Invalid XRPL address format"}

        response_data = {"tokens": [], "amm_lp_tokens": []}

        # Fetch regular tokens (trust lines)
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

        # Fetch AMM LP tokens
        account_objects_request = AccountObjects(account=address, type=AccountObjectType.AMM)
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
    """API endpoint to fetch token balances and values for an XRPL wallet address."""
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
