import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import sys
import pathlib2 as path
import os

# starts connection with Alpaca
key_id = os.environ['APCA_API_KEY_ID']
secret_key = os.environ['APCA_API_SECRET_KEY']
api = tradeapi.REST(key_id, secret_key)


'''
How to Run:
Given Algo sets of ticker pairs with total equity amount that each pair can trade with
The Algo figures outs what position for each pair it should take then sends the order
All closing price data throughout this file is assumed to be from oldest to most current
'''
# TODO Get market data for the desired pairs ... done
# TODO Determine which of the stocks should be shorted and which should be longed
# TODO Get account status
# TODO figure out how much money to allocate to each pair .... done
# TODO some stocks might be part of multiple pairs
# TODO buy/sell positions to match up with calculated, desired positions
# TODO cleanup?

'''
input should be a list of tuples, where each tuple contains the two symbols and an optional fraction of the total
available equity that should be used for that pair
If total_available_equity_limit = None, then total available equity will be calculated from the account's information
Input Example: pairs = [('AAPL', 'MSFT'), ('AAPL', 'NVDA', .5),('TLSA', 'FD', .1)]
This means that the first tuple would receive 40% of the total available equity, 
the second pair 50%, and the third pair, 10%
'''
def trade_pairs(pairs: list, total_available_equity_limit=None) -> None:
    # The key is the Stock Symbol, and the value is the occurrence.
    # This will be helpful later on for allocating money with stocks that are part of multiple pairs
    symbols_list = list(map(lambda x: x[0], pairs)) + list(map(lambda x: x[1], pairs))
    # casted into set momentarily to remove duplicates
    data = get_market_data(list(set(symbols_list)), 30)
    # returns the position that should be taken in each pair
    pair_positions = get_pairs_position(pairs, data)


# returns the calculated pair positions
# Symbol1, Symbol2, SymbolToLong
def get_pairs_position(pairs, data: dict, average_window=5, wide_spread=1.1, tight_spread=0.9):
    pair_positions = []
    for pair in pairs:
        symbol1 = pair[0]
        symbol2 = pair[1]
        average1 = np.mean(data[symbol1][-average_window:])
        average2 = np.mean(data[symbol2][-average_window:])
        spread_avg = (average1 - average2)
        curr1 = data[symbol1][-1]
        curr2 = data[symbol2][-1]
        spread_curr = (curr1 - curr2)

        if spread_curr > spread_avg * 1.10:  # detect a wide spread
            # short high, long low
            pair_positions.append(tuple(symbol1, symbol2, symbol1))

        elif spread_curr < spread_avg * .90:  # detect a tight spread
            # short low, long high
            pair_positions.append(tuple(symbol1, symbol2, symbol2))
        else:
            pair_positions.append(tuple(symbol1, symbol2, "None"))

    return pair_positions

    # TODO how to guarantee which one needs to be shorted and longed
    # TODO how to send back the info

def get_market_data(symbols: list, window) -> dict:
    batch_max = 200
    data = {}
    index = 0
    while index < len(symbols):
        symbol_batch = symbols[index:batch_max+index]

        # Retrieve stock data for current symbol batch
        barset = api.get_barset(
            symbols=symbol_batch,
            timeframe='day',
            limit=window,
        )

        for symbol in symbol_batch:
            bars = barset[symbol]
            closing_prices = list(map(lambda x: x.c, bars))
            symbol_name = str(symbol)
            data[symbol_name] = closing_prices

        index += batch_max

    return data

# reads in saved pairs and their equity allocations
# checks for valid symbols, percentages, etc.
def read_current_pairs_file():
    all_correct = True
    pairs = []
    with open("current_pairs.txt", 'r') as pairs_file:
        lines = pairs_file.readlines()

    # check that all symbols are legit
    assets = api.list_assets()
    symbols = set(map(lambda x: x.symbol, assets))

    # quick check if given equities are equal to or less than 100%
    # equities can be either out of 1 or 100. As long as they follow the same convection, its fine
    total_decimal = 0.0
    total_percent = 0.0
    all_decimals = False
    all_percentages = False
    pairs_without_allotment = 0

    for line in lines:

        try:
            if line == "\n":
                continue
            line = line.split(",")
            if line[0] == "#":
                continue
            if "" in line:
                line.remove("")
            if "\n" in line:
                line.remove("\n")
            # has a equity allocation

            # check symbols for accuracy
            if line[1] not in symbols:
                print(f"{line[1]} not found in ALPACA's list of tradable symbols")
                all_correct = False
            if line[0] not in symbols:
                print(f"{line[0]} not found in ALPACA's list of tradable symbols")
                all_correct = False

            allocation = float(line[2])
            if allocation <= 1:
                total_decimal += allocation
                all_decimals = True
            if allocation > 1:
                total_percent += allocation
                all_percentages = True
                line[2] = float(line[2]) / 100  # setting all given allocation to decimal scale

            pairs.append(tuple(line[:2] + [float(line[2])]))

        except IndexError:
            pairs_without_allotment += 1
            pairs.append(tuple(line[:2]))
            continue
        except ValueError:
            print(f"Incorrect number type in third position. Found {line[2]} instead.")
            sys.exit(-1)

    if total_percent > 100:
        print("Total given equity allocations are greater than 100%")
        sys.exit(-1)
    if total_decimal > 1.0:
        print("Total given equity allocations are greater than 1.0")
        sys.exit(-1)
    if all_decimals and all_percentages:
        print("Given equity allocations are in decimal and percentage forms")
        sys.exit(-1)

    # give remaining equity allotments to those that weren't given any
    if all_percentages:
        total_decimal = total_percent / 100

    if total_decimal == 1.0:
        pass
    else:
        remaining_allotment = 1.0 - total_decimal
        default_allotment = round(remaining_allotment / pairs_without_allotment, 3)
        for i in range(len(pairs)):
            if len(pairs[i]) == 2:
                pairs[i] = tuple([pairs[i][0], pairs[i][1], default_allotment])

    # print(pairs)
    if all_correct:
        print(pairs)
        return pairs
    else:
        print("Check the symbols found in current_pairs.txt")
        sys.exit(-1)




USAGE = "USAGE: Run from dir 'pairs_trading'\nOptions:\n" \
        "\tpython trade_pairs.py --> Run with all available account equity using saved pairs in 'current_pairs.txt'\n" \
        "\tpython trade_pairs.py -e INTEGER --> Run with given dollar limit using saved pairs\n" \
        "\tpython trade_pairs.py -help --> Gives list of available commands\n"

def valid_input():
    print(USAGE)


if __name__ == "__main__":
    try:
        print(sys.argv)
        if len(sys.argv) == 1:  # no args
            trade_pairs(read_current_pairs_file())
        elif sys.argv[1] == '-e':  # equity limit given
            trade_pairs(read_current_pairs_file(), int(sys.argv[2]))
        elif sys.argv[1] == '-help':
            valid_input()
        else:
            valid_input()

    except IndexError:
        print("Incorrect arguments given")
        valid_input()

    except ValueError:
        print("Incorrect arguments given")
        valid_input()

