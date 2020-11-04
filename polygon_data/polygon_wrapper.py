from polygon import RESTClient
import os
from datetime import datetime, timedelta
from requests import exceptions
import pandas as pd
import time

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
        self.__alpaca_key = os.environ[self.poly_api_key_name]

        # the polygon will only allow for 5000 items in a response. If asking for minute data, 5 days may result in
        # about 5000 data points. If asking for a larger time range than this 5000 data point limit, then the beginning
        # date will be what the result is calculated on. If asking for a range of data, doesn't include the
        # day at the start of the range
        with RESTClient(self.__alpaca_key) as client:
            try:
                resp = client.reference_tickers(market="stocks", active=True)
                if len(resp.results) == 0:
                    print("No result matching the given parameters.")

            except exceptions.HTTPError as e:
                if "401" in e.args[0]:
                    print("Unauthorized Key -> ", str(e))

                    # raising an error inside the __init__ will stop any object creation and or data leakage.
                    raise exceptions.HTTPError

    def get_minute_range_data(self, ticker_name: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        This is capable of return a large pandas dataframe of minute data of a single stock/ equity
        :param ticker_name: A valid ticker name. This is case sensitive. When in doubt, double check the spelling
        using the method client.reference_tickers(). Calling this method once over a time range of one month takes
        about 1.7 seconds on my computer.
        :param from_date: a starting date in the form YYYY-MM-DD
        :param to_date: an ending date in the form YYYY-MM-DD
        :return: pandas dataframe
        """
        print(f"Getting the data for {ticker_name} from {from_date} to {to_date}")
        start_date = datetime.strptime(from_date, "%Y-%m-%d")
        end_date = datetime.strptime(to_date, "%Y-%m-%d")
        print(f"Estimated time: {round((end_date - start_date).days * 1.7 / 30, 2)} seconds.")
        repeat = 1
        days_change = 5
        # more than 5 days of minute data may be more than 5000 data points and polygon wont send more than that
        # so i need to break up the range in to bite sizes chunks then append them together locally
        if (end_date - start_date).days > days_change:
            repeat = (end_date - start_date).days // days_change
            mid_date = start_date + timedelta(days=days_change)
        else:
            mid_date = end_date

        start_time = time.time()
        df = None
        with RESTClient(self.__alpaca_key) as client:
            for i in range(repeat):
                resp = client.stocks_equities_aggregates(ticker_name, 1, "minute",
                                                         start_date.strftime("%Y-%m-%d"), mid_date.strftime("%Y-%m-%d"))
                # resp.results is a list of dictionaries, with each dictionary representing a day

                # changes 't' from miliseconds since 1970 to datetime in isoformat for clarity
                for result in resp.results:
                    result['t'] = datetime.fromtimestamp(result["t"] / 1000).isoformat()

                temp_df = pd.DataFrame.from_dict(resp.results)
                if df is None:
                    df = temp_df
                else:
                    df = pd.concat([df, temp_df])

                # change range to request to polygon
                start_date = mid_date + timedelta(days=1)  # avoid getting the same data
                mid_date = start_date + timedelta(days=days_change)

                # edge cases
                if mid_date > end_date:
                    mid_date = end_date
                if start_date > end_date:
                    start_date = end_date

        print(f"time to complete is {time.time() - start_time} s")
        df.reset_index(drop=True, inplace=True)
        return df

# p = Poly()
# print(p.get_minute_range_data("AAPL", "2019-01-01", "2020-11-01"))
# str(p)

