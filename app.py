from flask import Flask, request, jsonify
from flask_cors import CORS
import xrpl
from xrpl.clients import AsyncJsonRpcClient
from xrpl.models.requests import AccountTx, AccountLines, BookOffers, AMMInfo
from datetime import datetime, timedelta
from collections import deque
import logging
import requests
import binascii
import asyncio
import aiohttp

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# XRPL client setup
JSON_RPC_URL = "https://s1.ripple.com:51234/"
client = AsyncJsonRpcClient(JSON_RPC_URL)

# DEX Screener API setup
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex/search"

# Cache to store frequently accessed token prices
price_cache = {}

def decode_hex_currency(hex_code):
    """Decode a 40-character hex currency code to ASCII."""
    if len(hex_code) == 40 and all(c in '0123456789ABCDEFabcdef' for c in hex_code):
        try:
            bytes_code = binascii.unhexlify(hex_code)
            return bytes_code.split(b'\0', 1)[0].decode('ascii')
        except Exception:
            pass
    return hex_code

async def decode_currency(currency, issuer):
    """Decode currency and identify AMM LP tokens using AMMInfo."""
    if len(currency) == 40 and all(c in '0123456789ABCDEFabcdef' for c in currency):
        try:
            amm_info = await client.request(AMMInfo(amm_account=issuer))
            required_fields = ["lp_token", "amount", "amount2"]
            if all(field in amm_info["amm"] for field in required_fields):
                asset1 = amm_info["amm"]["amount"]
                asset2 = amm_info["amm"]["amount2"]
                asset1_str = "XRP" if isinstance(asset1, str) else decode_hex_currency(asset1["currency"])
                asset2_str = "XRP" if isinstance(asset2, str) else decode_hex_currency(asset2["currency"])
                return f"LP_{asset1_str}_{asset2_str}", True
            else:
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
    return changes

async def fetch_dexscreener_price(decoded_currency):
    """Fetch token price from DEX Screener API using async requests."""
    if decoded_currency in price_cache:
        return price_cache[decoded_currency]

    url = f"{DEXSCREENER_API_URL}?q={decoded_currency}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    for pair in data.get('pairs', []):
                        if (pair.get('chainId') == 'xrpl' and
                                pair.get('baseToken', {}).get('symbol') == decoded_currency and
                                pair.get('quoteToken', {}).get('symbol') == 'XRP'):
                            price = float(pair.get('priceNative', '0'))
                            price_cache[decoded_currency] = price
                            return price
        logger.debug(f"No matching pairs found for {decoded_currency} on DEX Screener.")
        return None
    except asyncio.TimeoutError:
        logger.error(f"DEX Screener API timeout for {decoded_currency}")
        return None
    except Exception as e:
        logger.error(f"DEX Screener API request error: {e}")
        return None

async def get_dex_price(currency, issuer):
    """Fetch price from XRPL DEX."""
    try:
        buy_offers = await client.request(BookOffers(
            taker_pays={"currency": "XRP"},
            taker_gets={"currency": currency, "issuer": issuer},
            limit=10
        ))
        sell_offers = await client.request(BookOffers(
            taker_gets={"currency": "XRP"},
            taker_pays={"currency": currency, "issuer": issuer},
            limit=10
        ))

        MIN_OFFERS, MIN_VOLUME = 2, 10000
        buy_price = None
        if len(buy_offers["offers"]) >= MIN_OFFERS:
            xrp = sum(float(o["TakerPays"]) / 1_000_000 for o in buy_offers["offers"])
            tokens = sum(float(o["TakerGets"]["value"]) for o in buy_offers["offers"])
            if tokens >= MIN_VOLUME:
                buy_price = xrp / tokens
        sell_price = None
        if len(sell_offers["offers"]) >= MIN_OFFERS:
            xrp = sum(float(o["TakerGets"]) / 1_000_000 for o in sell_offers["offers"])
            tokens = sum(float(o["TakerPays"]["value"]) for o in sell_offers["offers"])
            if tokens >= MIN_VOLUME:
                sell_price = xrp / tokens
        if buy_price and sell_price:
            return (buy_price + sell_price) / 2
        return buy_price or sell_price or 0.000001
    except Exception as e:
        logger.error(f"XRPL DEX price error for {currency}-{issuer}: {e}")
        return 0.000001

@app.route('/token_pnl', methods=['POST'])
async def get_token_pnl():
    """Calculate token PNL, separating AMM LP tokens."""
    data = request.json
    address = data.get('address')

    if not address:
        return jsonify({'error': 'Wallet address is required'}), 400

    try:
        # Fetch transactions with pagination
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
            response = await client.request(req)
            result = response.result
            transactions.extend(result.get('transactions', []))
            marker = result.get('marker')
            if not marker:
                break

        # Fetch current holdings
        req = AccountLines(account=address)
        response = await client.request(req)
        lines = response.result['lines']
        holdings = {f"{line['currency']}-{line['account']}": float(line['balance'])
                    for line in lines if float(line['balance']) > 0.001}  # Filter dust

        xrp_balance = sum(float(line['balance']) for line in lines if line['currency'] == 'XRP')

        return jsonify({
            'xrp_balance': xrp_balance,
            'tokens': [],
            'amm_lp_tokens': []
        })
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
