import websocket
import json

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

trading_client = TradingClient('api-key', 'secret-key', paper=True)

# preparing market order
market_order_data = MarketOrderRequest(
                    symbol="SPY",
                    qty=0.023,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                    )

# Market order
market_order = trading_client.submit_order(
                order_data=market_order_data
               )

# preparing limit order
limit_order_data = LimitOrderRequest(
                    symbol="BTC/USD",
                    limit_price=17000,
                    notional=4000,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.FOK
                   )

# Limit order
limit_order = trading_client.submit_order(
                order_data=limit_order_data
              )

# Set your API key and secret key
API_KEY = ""
API_SECRET_KEY = ""


trading_client = TradingClient(API_KEY, API_SECRET_KEY)

# Get our account information.
account = trading_client.get_account()

# Check if our account is restricted from trading.
if account.trading_blocked:
    print('Account is currently restricted from trading.')

# Check how much money we can use to open new positions.
print('${} is available as buying power.'.format(account.buying_power))

def purchase_stock(
    ticker_symbol : str,
    quantity : float = 1.0, 
    limit_price : float = 0,
):
    if limit_price != 0:
        market_order_data = LimitOrderRequest(
            symbol=ticker_symbol,
            limit_price=limit_price,
            quantity=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
    else:
        market_order_data = MarketOrderRequest(
            symbol=ticker_symbol,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

    # Market order
    market_order = trading_client.submit_order(
        order_data=market_order_data
    )

    return True

def sell_stock(
    ticker_symbol : str,
    limit_price : float = 0,
):
    return True

def check_holdings():
    return True