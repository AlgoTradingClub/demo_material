# Pair Trading

## What' here?
*find_all_pairs.py makes a csv file that has the pairs that have certain criteria. These pairs will be used to pairs trade

## Find all pair
A smaller cointegration value will lead to to stocks that might follow each other too closely to generate any buy/sell signals. Consider chosing stocks that also have a high (>1.0) beta rating. This beta rating is made by comparing the stocks's long term beta to its short term beta. A beta rating of greater than 1 will mean that its been recency volatile

## trade_pairs
This file is meant to be run daily and readjusts the pairs positions. It goes off the data found in current_pairs.txt as how to allocate funds and know what pairs are currently being traded.
