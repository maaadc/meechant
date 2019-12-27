import requests
import os
import pandas as pd
import numpy as np


class MarketData:
    '''Querying and caching of stock market data. Currently based on data from alphavantage.co'''

    _cache_file = './marketdata_cache.h5'
    _api_key = os.getenv('ALPHAVANTAGE_API_KEY')

    def __init__(self):
        pass

    def get_daily(self, symbol):
        # For rapid development we cache the data in a HDF5 file. Yet, this method has some drawbacks.
        # Need to add a slash in front of the key and replace the ^ to suppress a warning
        store = pd.HDFStore(self._cache_file)
        symbol_key = '/' + symbol.replace('^', 'hat')
        if symbol_key not in store.keys():
            print(f'Retrieving {symbol}')
            data = self.request_data('TIME_SERIES_DAILY', symbol)
            store.put(key=symbol_key, value=data, append=False)
        data = store.get(symbol_key)
        store.close()
        return data

    def request_data(self, function_string, symbol):
        # Note: This function only implements daily time series currently
        if function_string != 'TIME_SERIES_DAILY':
            raise NotImplementedError
        response_data_key = 'Time Series (Daily)'
        response_columns_mapper = {'index': 'date',
                                   '1. open': 'open',
                                   '2. high': 'high',
                                   '3. low': 'low',
                                   '4. close': 'close',
                                   '5. volume': 'volume'}
        # For security reasons the api key is supplied via
        if len(os.getenv('ALPHAVANTAGE_API_KEY')) <= 1:
            raise AttributeError(
                'AlphaVantage API key not found in environment variables.')
        # Carry out query
        response = requests.get('https://www.alphavantage.co/query',
                                params={
                                    'function': function_string,
                                    'symbol': symbol,
                                    'outputsize': 'full',
                                    'apikey': os.getenv('ALPHAVANTAGE_API_KEY')
                                })
        res_json = response.json()
        if 200 == response.status_code and response_data_key in res_json:
            # todo: also handle meta data
            # read data frame from json response
            dfn = pd.DataFrame.from_dict(
                res_json[response_data_key], orient='index')
            # the date is given as unnamed index, create a new index
            dfn.reset_index(inplace=True)
            # rename columns, add symbol column
            dfn = dfn.rename(columns=response_columns_mapper)
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
