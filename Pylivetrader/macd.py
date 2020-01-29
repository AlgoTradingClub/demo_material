"""
MACD strategy for tech stocks
Enter "pylivetrader run macd.py --backend-config config.yaml" in the terminal to run
"""

# Needed for pylivetrader builtins (symbol, initialize, before_trading_start, context, data)
from pylivetrader.api import *
# Tabular data manipulator
import pandas as pd
# Regular alpaca API for calls to Polygon
import alpaca_trade_api as tradeapi
# For retrieval of environment variables (alpaca keys)
import os
# Efficient array operations
import numpy as np
# Needed for connecting to gmail
import smtplib
# Used for composing email
from email.message import EmailMessage
# Module for manipulating dates and times
from datetime import datetime

# Connect to Alpaca API
key_id = os.environ['alpaca_key_id']    # Check "Getting Started with Zipline" to see how to set environ vars in conda
secret_key = os.environ['alpaca_secret_key']
api = tradeapi.REST(key_id, secret_key)

# Setup smtp server
HOST = 'smtp.gmail.com'
PORT = '587'
server = smtplib.SMTP(host=HOST, port=PORT)
server.starttls()
SENDER = os.environ['sender']
EMAIL_PASSWORD = os.environ['email_password']
RECIPIENT = os.environ['recipient']
server.login(SENDER, EMAIL_PASSWORD)

# Used in email
ALGO_NAME = 'MACD'

# Called once at start of script
def initialize(context):
    # These are the assets that I'll be trading
    context.assets = symbols('AAPL', 'TSLA', 'GOOG', 'NFLX', 'AMZN', 'NVDA')


# Called once per day before market opens
def before_trading_start(context, data):
    # Compute asset weights relative to my portfolio value
    marketcaps = np.array([api.polygon.company(symbol.symbol).marketcap for symbol in context.assets])
    market_contribs = marketcaps / marketcaps.sum()

    context.asset_weights = dict(zip(context.assets, market_contribs))


# Called every time new data is received
def handle_data(context, data):
    # Moving average periods
    short_periods = 12
    long_periods = 26
    signal_periods = 9

    frequency = '1h'

    for symbol in context.assets:
        # Get data
        short_data = data.history(
            symbol, 'price', bar_count=short_periods, frequency=frequency)
        # Calculate exponential moving average (weights recent prices higher)
        short_ema = pd.Series.ewm(short_data, span=short_periods).mean()
        long_data = data.history(
            symbol, 'price', bar_count=long_periods, frequency=frequency)
        long_ema = pd.Series.ewm(long_data, span=long_periods).mean()

        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_periods).mean()

        # Trading logic
        action = None
        if macd.iloc[-1] > signal.iloc[-1]:
            # Adjusts position in {symbol} to {context.asset_weights[symbol} percent of current portfolio value
            order_target_percent(symbol, context.asset_weights[symbol])
            action = 'BUY'
        elif macd.iloc[-1] < signal.iloc[-1]:
            order_target_percent(symbol, 0)
            action = 'SELL'

        price = data.current(symbol, 'price')

        # If an order was submitted, send an email notifying me
        if action:
            notify(context, action, symbol, price, RECIPIENT)


def notify(context, action, symbol, price, recipient):
    now = datetime.now()
    time = now.strftime('%H:%M:%S')
    date = now.strftime('%m/%d/%Y')

    msg = EmailMessage()

    content = ''
    content += '%s order submitted on %s at %s for %s at $%.2f\n\n' % (action, date, time, symbol.symbol, price)
    content += 'Current portfolio:\n'
    for _, position in context.portfolio.positions.items():
        content += '%s - %s shares bought at %s\n' % (position.asset.symbol, position.amount, position.cost_basis)
        content += 'Current price: %s - Profit: %f\n' % (position.last_sale_price, (position.last_sale_price - position.cost_basis) * position.amount)
    content += 'Total value: %f' % context.portfolio.portfolio_value

    msg.set_content(content)
    msg['From'] = SENDER
    msg['To'] = recipient
    msg['Subject'] = ALGO_NAME + ' - trade notification'

    server.send_message(msg)