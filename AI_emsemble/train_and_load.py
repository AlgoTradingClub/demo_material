import pickle
import numpy as np
# import tensorflow as tf
# import tflearn
# from tflearn.layers.core import input_data, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.estimator import regression
from polygon_data import polygon_wrapper
import pandas as pd
from datetime import datetime, timedelta


# poly_api = polygon_wrapper.Poly()
# spy_data = poly_api.get_minute_range_data("SPY", "2019-01-01", "2020-10-30")

def label_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the data from the api and adds a new column, action. This looks at the yesterday and if the closing price is
    higher than today's closing price, it puts a '-1' (sell) in the action column. If yesterday's closing price is
    lower that today's closing price, it puts a '1' (buy) in the action column. This labels the data and allows for
    reinforcement learning for the nets and forests.
    :param data: A pandas dataframe with h (high), l (low), o (open), c (close), t (datetime), v (volume) columns.
    Assumes that the column t is sorted in either ascending or descending order
    :return: A pandas data frame with two columns, a 'id' and an 'action' column. The 'id' will match the respective
    id row of the input dataframe.
    """

    decending = False
    test = data[:2]["c"].to_list()
    test2 = data[:2].sort_values("t")["c"].to_list()
    if test == test2:
        decending = True
    else:
        # the data is in ascending order
        data = data[::-1]

    actions = np.zeros((data.axes[0].size,), dtype=int)
    index = 0
    previous = 0.0
    for row in data['c']:
        # Not taking into account the first day that has no yesterday reference
        if previous != 0.0:
            # if today is higher than yesterday, the system should have bought
            if row >= previous:
                actions[index] = 1
            # if today is lower than yesterday, the system should have sold
            else:
                actions[index] = -1
        previous = row
        index += 1

    data['action'] = actions
    if not decending:
        # flip the data frame back into the ascending order it came
        return data[::-1]

    return data


def make_cnn_model():
    """
    Defines the CNN model structure and returns the model
    :return: TFLearn Deep Neural Network
    """
    pass


# for validating the function `label_data()`

baseprice = np.random.normal(100.00, 10.0, 10)

test_data = {"c": baseprice, "t": []}
for i in range(10):
    today = datetime.now()
    # decending order
    new_datetime = today - timedelta(minutes=i)
    # acending order
    # new_datetime = today + timedelta(minutes=i)

    dt = new_datetime.isoformat()
    test_data['t'].append(dt)

test_df = pd.DataFrame(test_data)
convert_dict = {"c": float, "t": str, "action": int}

df = label_data(test_df)
print(df)