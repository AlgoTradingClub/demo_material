import numpy as np
import statsmodels.tsa.stattools as ts
import pathlib2 as path
import alpaca_trade_api as tradeapi
import pandas as pd
from pytz import timezone
import datetime
import os


# TODO 1 get all data
# TODO compare all data against each other, O(n^2);tr
# TODO Save and turn into DF and save. csv?

'''
Gets the correlation between the two times series sets of data.
Be aware, that np.corrcoef return a 2x2 array, from which there are two duplicate answers that are non-unity.
i.e. coorcoef returns:
[[1.0,.848684]
[.848684, 1.0]]
.848684 is the correlation coefficient and 1's can be ignored
'''
def correlated(stock1, stock2) -> float:
    corr_arr = np.corrcoef(stock1, stock2)
    if corr_arr[0][1] != 1.0:
        return corr_arr[0][1]
    else:
        return corr_arr[0][0]


'''
Returns the conintegration value of the two time-series data sets
Slower operation than correlation
'''
def cointegrated(stock1, stock2) -> float:
    return ts.coint(stock1, stock2)


def calc_beta(bars, benchmark_bars):
    cov = np.cov(bars, benchmark_bars)[0][1]
    var = np.var(benchmark_bars)
    beta = cov / var

    return beta


'''
Finds all cointegrated stocks within a threshold.
Returns cointegrated pairs with their coint value and short and long-term beat value
if limit == -1, will calculate all available stocks.
'''
def find_all_pairs(limit=-1, coint_value=0.01, coor_value=0.85, short_term_beta_window=30, long_term_beta_window=90):
    # starts connection with Alpaca
    key_id = os.environ['APCA_API_KEY_ID']
    secret_key = os.environ['APCA_API_SECRET_KEY']
    api = tradeapi.REST(key_id, secret_key)

    # set up pandas dataframe for data management later
    ratings = pd.DataFrame(columns=['Symbol_1', 'Symbol_2', 'Coint_Rating', 'Price Symbol_1', 'Price Symbol_2',
                                    'Symbol_1 Beta Rating', 'Symbol_2 Beta Rating'])

    # gets all the assets available to Alpaca and creates a list of all or up to a prescribed limit
    assets = api.list_assets()
    if limit < 0 or limit > len(assets):
        assets = [asset for asset in assets if asset.tradable and asset.shortable]
    else:
        assets = [assets[i] for i in range(limit) if assets[i].tradable and assets[i].shortable]

    index = 0
    batch_size = 200  # Max number of stocks per api request

    # Get SPY data to use as market component of beta calc
    SPY_bars = api.get_barset(
        symbols=['SPY'],
        timeframe='day',
        limit=long_term_beta_window
    )['SPY']

    closing_spy = list(map(lambda x: x.c, SPY_bars))
    short_term_closing_spy = list(map(lambda x: x.c, SPY_bars[-short_term_beta_window:]))

    all_bars = dict()

    while index < len(assets):
        print(f"index {index} of {len(assets)}")  # to track progress

        symbol_batch = [asset.symbol for asset in assets[index:index + batch_size]]

        # Retrieve stock data for current symbol batch
        barset = api.get_barset(
            symbols=symbol_batch,
            timeframe='day',
            limit=long_term_beta_window,
        )

        for symbol in symbol_batch:
            bars = barset[symbol]
            closing_prices = list(map(lambda x: x.c, bars))
            all_bars[symbol] = closing_prices

        index += batch_size

    # now all closing price data is saved and ready to be further analyzed
    # TODO rename a and b into something better
    for a in all_bars:
        for b in all_bars:
            if a != b:  # don't compare a stock against itself
                print(f"{a}: len-> {len(all_bars[a])}, {b}: len-> {len(all_bars[b])}")
                if correlated(all_bars[a], all_bars[b]) > coor_value:
                    cointegration_value = cointegrated(all_bars[a], all_bars[b])
                    if cointegration_value < coint_value:  # its cointegrated!!

                        long_term_beta_a = calc_beta(
                                    bars=all_bars[a],
                                    benchmark_bars=closing_spy,
                                )
                        short_term_beta_a = calc_beta(
                            bars= all_bars[a][-short_term_beta_window:],
                            benchmark_bars=short_term_closing_spy
                        )
                        rating_a = short_term_beta_a / long_term_beta_a

                        long_term_beta_b = calc_beta(
                            bars=all_bars[b],
                            benchmark_bars=closing_spy,
                        )
                        short_term_beta_b = calc_beta(
                            bars=all_bars[b][-short_term_beta_window:],
                            benchmark_bars=short_term_closing_spy
                        )
                        rating_b = short_term_beta_b / long_term_beta_b

                        ratings = ratings.append({
                                        'Symbol_1': a,
                                        'Symbol_2': b,
                                        'Coint_Rating': cointegration_value,
                                        'Price Symbol_1': all_bars[a][-1].c,
                                        'Price Symbol_2': all_bars[b][-1].c,
                                        'Symbol_1 Beta Rating': rating_a,
                                        'Symbol_2 Beta Rating': rating_b
                                    }, ignore_index=True)


            # if len(bars) == long_term_beta_window:
            #
            #     # Rate stocks based on beta
            #     # rating = short_term_beta / long_term_beta
            #     closing_prices = list(map(lambda x: x.c, bars))
            #     long_term_beta = calc_beta(
            #         bars=closing_prices,
            #         benchmark_bars=closing_spy,
            #     )
            #     short_term_closing_prices = list(map(lambda x: x.c, bars[-short_term_beta_window:]))
            #     short_term_beta = calc_beta(
            #         bars=short_term_closing_prices,
            #         benchmark_bars=short_term_closing_spy
            #     )
            #     rating = short_term_beta / long_term_beta
            #
            #     # if its been more volatile recently than 90 days ago, then append
            #     if rating > 1:
            #         ratings = ratings.append({
            #             'symbol': symbol,
            #             'rating': rating,
            #             'price': bars[-1].c
            #         }, ignore_index=True)

    ratings = ratings.sort_values('Coint_Rating', ascending=True)
    ratings = ratings.reset_index(drop=True)

    results_dir = path.Path.joinpath(path.Path.cwd(), "pairs_trading", "results")
    with open(f"pairs_results_{datetime.date.today()}.csv", "w+") as f:
        f.write(ratings.to_csv())


find_all_pairs(200)
