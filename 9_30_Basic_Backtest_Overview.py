# Get historical data from polygon through python alpaca API wrapper
import alpaca_trade_api as tradeapi
from keys import public_key, secret_key
# Authenticate alpaca account

client = tradeapi.REST(public_key, secret_key,'https://paper-api.alpaca.markets')
# Collect data for aapl from the past year.
AAPL_data = client.polygon.historic_agg('day', 'AAPL', '2019-1-1', '2019-9-23').df

# Clean Data
closing_prices = AAPL_data.close

#-------------------------------------------------------------#


# mean reversion strategy backtest

# Define important variables
mva_period = 30
pct_deviation = 0.05

last_qty = 0.
last_price = 0.

cash = 100000.
history = [cash]

# enter simulated trading loop
for i in range(mva_period,len(closing_prices)):
  p = closing_prices[i]
  mva = 0.0
  #calculate moving average
  for x in range(1,mva_period+1):
    mva+=closing_prices[i-x]/mva_period

  # if it's low, buy it.
  if p < mva*(1-pct_deviation) and last_qty <= 0:
    cash += last_price*last_qty - last_qty*p
    last_price = p
    last_qty = cash//p

  # if it's high, sell it.
  elif p > mva*(1+pct_deviation) and last_qty >= 0:
    cash += last_qty*p - last_price*last_qty
    last_price = p
    last_qty = -cash//p
    #last_qty = 0

  history.append(cash)
  

import matplotlib.pyplot as plt
rolling_ave = closing_prices.rolling(mva_period)
rolling_ave.mean().plot()
closing_prices.plot()
plt.show()
plt.plot(history)
plt.show()
print(cash)


