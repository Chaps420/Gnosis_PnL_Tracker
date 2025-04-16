from flask import Flask, request, jsonify
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountLines, AccountObjects
from xrpl.models.requests.account_objects import AccountObjectType
import asyncio
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# XRPL client (use testnet or mainnet depending on your needs)
XRPL_CLIENT = JsonRpcClient("https://s1.ripple.com:51234/")  # Mainnet
# XRPL_CLIENT = JsonRpcClient("https://s.altnet.rippletest.net:51234/")  # Testnet

async def get_wallet_tokens(address):
    """Fetch all tokens (regular and AMM LP) held by the XRPL wallet."""
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
        account_lines_response = await XRPL_CLIENT.request(account_lines_request)
        
        if account_lines_response.is_successful():
            for line in account_lines_response.result.get("lines", []):
                token = {
                    "currency": line["currency"],
                    "issuer": line["account"],
                    "amount_held": float(line["balance"])  # Convert to float for JSON serialization
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
        account_objects_response = await XRPL_CLIENT.request(account_objects_request)

        if account_objects_response.is_successful():
            for obj in account_objects_response.result.get("account_objects", []):
                if obj.get("LedgerEntryType") == "AMM":
                    lp_token = {
                        "currency": obj.get("LPToken", {}).get("currency", "N/A"),
                        "issuer": obj.get("LPToken", {}).get("issuer", "N/A"),
                        "amount_held": float(obj.get("LPTokenBalance", {}).get("value", 0))
                    }
                    response_data["amm_lp_tokens"].append(lp_token)
        else:
            logger.error(f"Failed to fetch account objects: {account_objects_response.result}")
            return {"error": "Failed to fetch AMM LP tokens"}

        return response_data

    except Exception as e:
        logger.error(f"Error fetching wallet tokens: {str(e)}")
        return {"error": f"Server error: {str(e)}"}

@app.route('/token_pnl', methods=['POST'])
async def token_pnl():
    """API endpoint to fetch current token balances for a given XRPL wallet address."""
    try:
        data = request.get_json()
        address = data.get("address", "").strip()

        if not address:
            return jsonify({"error": "No address provided"}), 400

        # Fetch wallet tokens
        result = await get_wallet_tokens(address)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in token_pnl endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    # Run Flask app
    app.run(host="0.0.0.0", port=5000)
