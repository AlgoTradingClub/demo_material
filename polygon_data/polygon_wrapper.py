from polygon import RESTClient
import os
from datetime import datetime
from requests import exceptions
import pandas as pd

class Poly:
    """
    This is an example for what the polygon api can do
    See https://polygon.io/docs/ for all the available endpoints
    """
    # Edit this to the key name of the live alpaca key
    # to access polygon data, you must have at least 1 $ in a live alpaca account.
    poly_api_key_name = "APCA_API_KEY_ID_LIVE"

    def __init__(self):
        # this will get the computer's environment variable and load it in.
        # this is safely store alpaca keys
        alpaca_key = os.environ[self.poly_api_key_name]

        # the polygon will only allow for 5000 items in a response. If asking for minute data, 10 days will result in
        # about 5000 data points. If asking for a larger time range than this 5000 data point limit, then the beginning
        # date will be what the result is calculated on. If asking for a range of data, doesn't include the
        # day at the start of the range
        with RESTClient(alpaca_key) as client:
            try:
                resp = client.stocks_equities_aggregates("SPY", 1, "minute", "2020-01-01", "2020-01-03")
                if len(resp.results) != 0:
                    for result in resp.results:
                        result['t'] = datetime.fromtimestamp(result["t"] / 1000)
                    s = pd.DataFrame.from_dict(resp.results)
                    print(s)

                else:
                    print("No result matching the given parameters.")

            except exceptions.HTTPError as e:
                if "401" in e.args[0]:
                    print("Unauthorized Key -> ", str(e))



p = Poly()

