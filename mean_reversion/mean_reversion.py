import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import sys
import logging

from datetime import datetime, timedelta
from pytz import timezone
from collections import defaultdict

logging.basicConfig(filename='mean_reversion.log', level=logging.INFO)

"""
1. Select stocks with high short-term beta and low long-term beta
2. Trade using mean-reversion
"""
stocks_to_trade = 20
short_term_beta_window = 30  # Days to use in short-term beta calculation
long_term_beta_window = 90  # Days to use in long-term beta calculation
mean_window = 10            # Used to calc rolling price mean
api_time_format = '%Y-%m-%dT%H:%M:%S.%f-04:00'


# TODO: change to calc just variance instead
def calc_beta(bars, benchmark_bars):
    cov = np.cov(bars, benchmark_bars)[0][1]
    var = np.var(benchmark_bars)
    beta = cov / var

    return beta


def get_ratings(algo_time, stocks_to_trade=stocks_to_trade):
    assets = api.list_assets()
    assets = [asset for asset in assets if asset.tradable and asset.shortable]
    ratings = pd.DataFrame(columns=['symbol', 'rating', 'price'])
    index = 0
    batch_size = 200    # Max number of stocks per api request
    formatted_time = None
    if algo_time is not None:
        # Convert algo_time to alpaca's format
        formatted_time = algo_time.date().strftime(api_time_format)

    # Get SPY data to use as market component of beta calc
    SPY_bars = api.get_barset(
        symbols=['SPY'],
        timeframe='day',
        limit=long_term_beta_window,
        end=formatted_time
    )['SPY']

    while index < len(assets):
        symbol_batch = [
            asset.symbol for asset in assets[index:index+batch_size]
        ]

        # Retrieve stock data for current symbol batch
        barset = api.get_barset(
            symbols=symbol_batch,
            timeframe='day',
            limit=long_term_beta_window,
            end=formatted_time
        )

        for symbol in symbol_batch:
            bars = barset[symbol]
            if len(bars) == long_term_beta_window:
                # Make sure the most recent data isn't missing
                latest_bar = bars[-1].t.to_pydatetime().astimezone(timezone('EST'))
                #gap_from_present = algo_time - latest_bar
                # if gap_from_present.days > 1:
                #     continue

                # Rate stocks based on beta
                # rating = short_term_beta / long_term_beta
                closing_prices = list(map(lambda x: x.c, bars))
                closing_SPY = list(map(lambda x: x.c, SPY_bars))
                long_term_beta = calc_beta(
                    bars=closing_prices,
                    benchmark_bars=closing_SPY,
                )
                short_term_closing_prices = list(map(lambda x: x.c, bars[-short_term_beta_window:]))
                short_term_closing_SPY = list(map(lambda x: x.c, SPY_bars[-short_term_beta_window:]))
                short_term_beta = calc_beta(
                    bars=short_term_closing_prices,
                    benchmark_bars=short_term_closing_SPY
                )
                rating = short_term_beta / long_term_beta
                if rating > 1:
                    ratings = ratings.append({
                        'symbol': symbol,
                        'rating': rating,
                        'price': bars[-1].c
                    }, ignore_index=True)
        index += batch_size

    ratings = ratings.sort_values('rating', ascending=False)
    ratings = ratings.reset_index(drop=True)

    return ratings[:stocks_to_trade]


def get_shares_to_buy(ratings_df, cash, portfolio_alloc=1):
    total_rating = ratings_df['rating'].sum()
    shares_dict = {}
    for _, row in ratings_df.iterrows():
        prc_to_alloc = row['rating'] / total_rating * portfolio_alloc
        amount = prc_to_alloc * cash
        shares = int(amount / row['price'])
        shares_dict[row['symbol']] = shares

    return shares_dict


def live_trade(api, portfolio_alloc, ratings):
    # Get updated barset for rated stocks
    daily_barset = api.get_barset(
        symbols=ratings['symbol'],
        timeframe='day',
        limit=mean_window,
        end=None
    )
    current_bars = api.get_barset(
        symbols=ratings['symbol'],
        timeframe='1Min',
        limit=3,
        end=None
    )

    # Update most recent prices for each stock in ratings
    for symbol, bars in current_bars.items():
        if len(bars) > 0:
            while bars[-1].c == np.nan:
                bars = bars[:-1]
            ratings.loc[ratings['symbol'] == symbol, 'price'] = bars[-1].c
        else:
            ratings.loc[ratings['symbol'] == symbol, 'price'] = np.nan

    # Calculate the number of each share to buy
    cash = float(api.get_account().cash)
    shares_to_buy = get_shares_to_buy(ratings, cash, portfolio_alloc)

    # Mean reversion logic
    for symbol, n_shares in shares_to_buy.items():
        bars = daily_barset[symbol]
        closing_prices = list(map(lambda x: x.c, bars))
        long_term_mean = np.nanmean(closing_prices)
        current_price = ratings.loc[ratings['symbol'] == symbol].iloc[0]['price']
        std_dev = np.nanstd(closing_prices)

        # TODO: check that no order is made when a pending order for the same symbol exists
        current_position = None
        try:
            current_position = api.get_position(symbol).side
        except:
            pass

        if current_price > long_term_mean + std_dev and current_position != 'short':
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=n_shares,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                logging.info(f'Sell order submitted for {n_shares} of {symbol} at {current_price} on {datetime.now(timezone("EST"))}')
            except Exception as err:
                logging.error(err)

        elif current_price < long_term_mean - std_dev and current_position != 'long':
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=n_shares,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                logging.info(f'Buy order submitted for {n_shares} of {symbol} at {current_price} on {datetime.now(timezone("EST"))}')
            except Exception as err:
                logging.error(err)
        else:
            pass


# Returns a string version of a timestamp compatible with the Alpaca API.
def api_format(dt):
    return dt.strftime(api_time_format)


def calc_portfolio_val(api, cash, shares):
    barset = api.get_barset(
        symbols=shares.keys(),
        timeframe='day',
        limit=3,
    )
    portfolio_value = 0
    for symbol, data in barset.items():
        portfolio_value += data[-1].c * shares[symbol]
    portfolio_value += cash

    return portfolio_value


def backtest(api, testing_days, starting_funds, backtest_name=''):
    cash = starting_funds
    shares = defaultdict(int)
    portfolio = pd.DataFrame(columns=['cash', 'symbol', 'qty', 'price', 'portfolio_value'])
    backtest_bars = pd.DataFrame(columns=['open', 'close', 'low', 'high', 'lt_mean'])

    # Set backtesting timeframe
    now = datetime.now(timezone('EST'))
    beginning = now - timedelta(days=testing_days)

    # Get list of trading days over backtesting timeframe
    trading_days = api.get_calendar(
        start=beginning.strftime('%Y-%m-%d'),
        end=now.strftime('%Y-%m-%d')
    )

    # Get ratings at the beginning of the testing window
    ratings = get_ratings(algo_time=beginning)
    for day in trading_days:
        # Get updated barset for rated stocks
        end_datetime = datetime.combine(day.date, day.open) + timedelta(hours=1)
        formatted_end = api_format(end_datetime)
        daily_barset = api.get_barset(
            symbols=ratings['symbol'],
            timeframe='day',
            limit=mean_window,
            end=formatted_end
        )
        current_bars = api.get_barset(
            symbols=ratings['symbol'],
            timeframe='1Min',
            limit=3,
            end=formatted_end
        )

        # Update most recent prices for each stock in ratings
        for symbol, bars in current_bars.items():
            if len(bars) > 0:
                while bars[-1].c == np.nan:
                    bars = bars[:-1]
                ratings.loc[ratings['symbol'] == symbol, 'price'] = bars[-1].c

                name = (symbol, end_datetime)
                backtest_bars = backtest_bars.append(
                    pd.Series(
                        {'open': bars[-1].o,
                         'close': bars[-1].c,
                         'low': bars[-1].l,
                         'high': bars[-1].h},
                        name=(symbol, end_datetime)
                    )
                )
            else:
                ratings.loc[ratings['symbol'] == symbol, 'price'] = daily_barset[symbol][-1].c
                name = (symbol, end_datetime)
                backtest_bars = backtest_bars.append(
                    pd.Series(
                        {'open': daily_barset[symbol][-1].o,
                         'close': daily_barset[symbol][-1].c,
                         'low': daily_barset[symbol][-1].l,
                         'high': daily_barset[symbol][-1].h},
                        name=(symbol, end_datetime)
                    )
                )

        # Calculate the number of each share to buy
        shares_to_buy = get_shares_to_buy(ratings, cash)

        # Mean reversion logic
        for symbol, n_shares in shares_to_buy.items():
            bars = daily_barset[symbol]
            closing_prices = list(map(lambda x: x.c, bars))
            long_term_mean = np.nanmean(closing_prices)
            # Record long term mean for this trade
            backtest_bars.at[(symbol, name[1]), 'lt_mean'] = long_term_mean

            current_price = ratings.loc[ratings['symbol'] == symbol].iloc[0]['price']
            std_dev = np.std(closing_prices)

            if current_price > long_term_mean + std_dev and shares[symbol] >= 0:
                shares[symbol] = -n_shares
                cash += n_shares * current_price
                portfolio = portfolio.append(
                    {'cash': cash,
                     'symbol': symbol,
                     'qty': -n_shares,
                     'price': current_price,
                     'portfolio_value': calc_portfolio_val(api, cash, shares),
                     'time': end_datetime},
                    ignore_index=True
                )
            elif current_price < long_term_mean - std_dev and shares[symbol] <= 0:
                shares[symbol] = n_shares
                cash -= n_shares * current_price
                portfolio = portfolio.append(
                    {'cash': cash,
                     'symbol': symbol,
                     'qty': n_shares,
                     'price': current_price,
                     'portfolio_value': calc_portfolio_val(api, cash, shares),
                     'time': end_datetime},
                    ignore_index=True
                )
            else:
                pass

    # Backtest results
    final_portfolio_value = calc_portfolio_val(api, cash, shares)
    print(f'Cash: {cash}')
    print(f'Positions: {shares}')
    print(f'Equity: {final_portfolio_value}')
    print(f'Return: {(final_portfolio_value / starting_funds - 1) * 100}%')

    # Print market (S&P500) return for the time period
    sp500_bars = api.get_barset(
        symbols='SPY',
        timeframe='day',
        start=api_format(trading_days[0].date),
        end=api_format(trading_days[-1].date)
    )['SPY']
    sp500_change = (sp500_bars[-1].c - sp500_bars[0].c) / sp500_bars[0].c
    print('S&P 500 change during backtesting window: {:.4f}%'.format(
        sp500_change * 100)
    )

    # Save backtest data
    portfolio.to_pickle(backtest_name+'portfolio.pkl')

    index = list(backtest_bars.index)
    backtest_bars.index = pd.MultiIndex.from_tuples(index, names=['symbol', 'time'])
    backtest_bars.to_pickle(backtest_name+'bars.pkl')

    ratings.to_pickle(backtest_name+'ratings.pkl')

    return final_portfolio_value


if __name__ == '__main__':
    api = tradeapi.REST()

    if len(sys.argv) < 2:
        print('Error: please specify a command;')
        print('Options:')
        print('backtest <cash balance> <number of days to test> <optional: path to save backtest results>')
        print('live <percent of portfolio to allocate ex. 1 for 100%> <stock rating path>')
        print('rerank <path to save stock rating data>')
    else:
        if sys.argv[1] == 'backtest':
            # Run a backtesting session using the provided parameters
            starting_funds = float(sys.argv[2])
            testing_days = int(sys.argv[3])

            if len(sys.argv) > 4:
                backtest_name = str(sys.argv[4])
            else:
                backtest_name = ''
            portfolio_value = backtest(api, testing_days, starting_funds, backtest_name)
            portfolio_change = (portfolio_value - starting_funds) / starting_funds
            print('Portfolio change: {:.4f}%'.format(portfolio_change*100))

        elif sys.argv[1] == 'rerank':
            # Rerank
            ratings_f_name = sys.argv[2]
            ratings = get_ratings(algo_time=None)
            ratings.to_pickle(ratings_f_name)

            # Cancel outstanding orders and close all positions
            api.cancel_all_orders()
            api.close_all_positions()

        elif sys.argv[1] == 'live':
            portfolio_alloc = sys.argv[2]
            try:
                portfolio_alloc = float(portfolio_alloc)
            except Exception as e:
                print(e)
                print(f'Error: Value should be between (0-1], got {portfolio_alloc}')

            if portfolio_alloc <= 0 or portfolio_alloc > 1:
                portfolio_alloc = 1

            ratings_f_name = sys.argv[3]
            ratings = pd.read_pickle(ratings_f_name)
            live_trade(api, portfolio_alloc, ratings=ratings)
        else:
            print('Error: Unrecognized command ' + sys.argv[1])
