from .config import alphavantage

import requests
import pandas as pd
import numpy as np


class MarketData:

    _cache = {}

    def __init__(self):
        pass

    def get_daily(self, symbol):
        if symbol not in list(self._cache.keys()):
            data = self.request_data('TIME_SERIES_DAILY', symbol)
            self._cache.update({symbol: data})
        return self._cache[symbol]

    def request_data(self, function_string, symbol):
        response = requests.get('https://www.alphavantage.co/query',
                                params={
                                    'function': function_string,
                                    'symbol': symbol,
                                    'outputsize': 'full',
                                    'apikey': alphavantage.secret_api_key
                                })
        res_json = response.json()
        if 200 == response.status_code and alphavantage.data_key_daily in res_json:
            # todo: also handle meta data
            # read data frame from json response
            dfn = pd.DataFrame.from_dict(
                res_json[alphavantage.data_key_daily], orient='index')
            # the date is given as unnamed index, create a new index
            dfn.reset_index(inplace=True)
            # rename columns, add symbol column
            dfn = dfn.rename(columns=alphavantage.columns_mapper)
            # data was read as strings, convert to suitable data types and sort
            dfn = dfn.astype({'open': float, 'high': float,
                              'low': float, 'close': float, 'volume': int})
            dfn['date'] = pd.to_datetime(dfn['date'])
            dfn.sort_values(by=['date'], inplace=True, ascending=True)
            # create index
            dfn.set_index(['date'], inplace=True)
            # add some meaningful feature
            dfn['price'] = 0.25 * \
                (dfn['open'] + dfn['high'] + dfn['low'] + dfn['close'])
            dfn['swing'] = (dfn['high'] - dfn['low']) / dfn['price']
            dfn['slope'] = (dfn['close'] - dfn['open']) / dfn['price']
            # print and return
            return dfn
        else:
            print(
                f'''Could not retrieve market data for "{symbol}" from "{response.url}".
                    Status code is {response.status_code} and response was: {res_json}''')
            return None


# Singleton
market = MarketData()
