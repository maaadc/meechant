'''Configuration variables'''


class alphavantage:
    secret_api_key = 'demo'
    data_key_daily = 'Time Series (Daily)'
    columns_mapper = {'index': 'date',
                      '1. open': 'open',
                      '2. high': 'high',
                      '3. low': 'low',
                      '4. close': 'close',
                      '5. volume': 'volume'}
