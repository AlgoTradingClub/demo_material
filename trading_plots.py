import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_mean_reversion():
    # Generate data
    days = np.arange(100)
    prices = np.array([np.sin(day) for day in days]) * 50 + 100 + np.random.normal(scale=5, size=len(days))

    # Trade
    y_std = np.std(prices)
    y_mean = np.mean(prices)

    sell_thresh = y_mean + .75*y_std
    buy_thresh = y_mean - .75*y_std
    
    profit = 0
    buys = []
    bought = False
    sells = []
    for day in range(len(days)):
        price = prices[day]

        # Buy
        if not bought and price < buy_thresh:
            bought = True
            buys.append((day, price))
        # Sell
        if bought and price > sell_thresh:
            bought = False
            sells.append((day, price))
            profit += price - buys[-1][1]

    # Plot
    plt.title(f'Mean Reversion: Profit ${profit:.2f}')
    plt.plot(days, prices, label='price')
    plt.xlabel('Day')
    plt.ylabel('$')
    plt.ylim(0, 250)
    plt.scatter([point[0] for point in buys], [point[1] for point in buys], label='Buy')
    plt.scatter([point[0] for point in sells], [point[1] for point in sells], label='Sell')
    plt.legend()
    plt.show()

def plot_macd():
    # Generate data
    days = np.arange(300)
    prices = [50]
    for _ in range(len(days)-1):
        prices.append(prices[-1] + np.random.normal(loc=0, scale=5))
        
    prices = pd.Series(prices)
        
    buys = []
    bought = False
    sells = []
    
    short_window = 12
    short_mavg = prices.ewm(span=short_window, min_periods=short_window).mean()
    
    long_window = 26
    long_mavg = prices.ewm(span=long_window, min_periods=long_window).mean()
    
    macd = short_mavg - long_mavg
    signal_window = 9
    signal = macd.ewm(span=signal_window, min_periods=signal_window).mean()
    
    # Trade
    profit = 0
    buys = []
    bought = False
    sells = []
    for day in range(len(signal)):
        # Buy
        if not bought and macd[day] > signal[day]:
            bought = True
            buys.append((day, prices[day]))
        
        # Sell
        if bought and macd[day] < signal[day]:
            bought = False
            sells.append((day, prices[day]))
            profit += prices[day] - buys[-1][1]
    
    # Plot
    plt.title(f'MACD: Profit ${profit:.2f}')
    plt.xlabel('Day')
    plt.ylabel('$')
    plt.plot(days, prices, label='price')
    # plt.plot(days, short_mavg, label='short_mavg')
    # plt.plot(days, long_mavg, label='long_mavg')
    plt.plot(days, macd, label='macd')
    plt.plot(days, signal, label='signal')
    plt.scatter([buy[0] for buy in buys], [buy[1] for buy in buys], label='Buy')
    plt.scatter([sell[0] for sell in sells], [sell[1] for sell in sells], label='Sell')
    plt.legend()
    plt.show()
    
def plot_pairs():
    # Generate data
    days = np.arange(200)
    prices1 = np.array([np.sin(day) for day in days]) + 50 + np.random.normal(scale=2, size=len(days))
    prices2 = prices1 + 50 + np.random.normal(scale=2, size=len(days))
    
    # Historic price ratio (first 50 days)
    price_ratio_mean = np.mean(prices2[:50] / prices1[:50])
    price_ratio_std = np.std(prices2[:50] / prices1[:50])
    long_stock1_thresh = price_ratio_mean + 14 * price_ratio_std
    long_stock2_thresh = price_ratio_mean - 14 * price_ratio_std
    
    # Stock 1 rises, stock 2 falls (days 50 - 100)
    # def parabola(x): return -(x / 5 - 5) ** 2 + 50
    def parabola(x): return (-((x/5)-5)**2)/2 + 20
    
    for day in range(50, 102):
        change_val = parabola(day - 50)
        prices1[day] += change_val
        prices2[day] -= change_val
    
    # Stock 1 falls, stock 2 rises (days 125 - 175)
    for day in range(125, 177):
        change_val = parabola(day - 125)
        prices1[day] -= change_val
        prices2[day] += change_val
    
    # Trade
    profit = 0
    buys1 = []
    buys2 = []
    close_buys1 = []
    close_buys2 = []
    shorts1 = []
    shorts2 = []
    close_shorts1 = []
    close_shorts2 = []
    long_stock1 = False
    long_stock2 = False
    
    for day in range(len(days)):
        curr_ratio = prices2[day] / prices1[day]
        
        # Long stock 1, short stock 2
        if not long_stock1 and curr_ratio > long_stock1_thresh:
            long_stock1 = True
            buys1.append((day, prices1[day]))
            shorts2.append((day, prices2[day]))
            
        # Short stock 1, long stock 2
        if not long_stock2 and curr_ratio < long_stock2_thresh:
            long_stock2 = True
            buys2.append((day, prices2[day]))
            shorts1.append((day, prices1[day]))
            
        # Close positions
        if (price_ratio_mean - 2 * price_ratio_std <= curr_ratio <= price_ratio_mean + 2 * price_ratio_std):
            if long_stock1:
                long_stock1 = False
                
                close_buys1.append((day, prices1[day]))
                profit += close_buys1[-1][1] - buys1[-1][1]
                
                close_shorts2.append((day, prices2[day]))
                profit += shorts2[-1][1] - close_shorts2[-1][1]
            if long_stock2:
                long_stock2 = False
                
                close_buys2.append((day, prices2[day]))
                profit += close_buys2[-1][1] - buys2[-1][1]
                
                close_shorts1.append((day, prices1[day]))
                profit += shorts1[-1][1] - close_shorts1[-1][1]
    
    # Plot
    plt.title(f'Pairs Trading: Profit ${profit:.2f}')
    plt.xlabel('Day')
    plt.ylabel('$')
    plt.plot(days, prices1, label='Stock A')
    plt.plot(days, prices2, label='Stock B')
    plt.scatter([buy[0] for buy in buys1], [buy[1] for buy in buys1], s=100, label='Buy A')
    plt.scatter([buy[0] for buy in buys2], [buy[1] for buy in buys2], s=100, label='Buy B')
    plt.scatter([short[0] for short in shorts1], [short[1] for short in shorts1], s=100, label='Short A')
    plt.scatter([short[0] for short in shorts2], [short[1] for short in shorts2], s=100, label='Short B')
    plt.scatter([sell[0] for sell in close_buys1], [sell[1] for sell in close_buys1], s=100, label='Sell A')
    plt.scatter([sell[0] for sell in close_buys2], [sell[1] for sell in close_buys2], s=100, label='Sell B')
    plt.scatter([close_short[0] for close_short in close_shorts1], [close_short[1] for close_short in close_shorts1], s=100, label='Close short A')
    plt.scatter([close_short[0] for close_short in close_shorts2], [close_short[1] for close_short in close_shorts2], s=100, label='Close short B')
    plt.legend()
    plt.show()

plot_mean_reversion()
plot_macd()
plot_pairs()