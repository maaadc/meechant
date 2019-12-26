from ..backtest import Transaction

import abc
from typing import List


class TradingStrategy(abc.ABC):
    '''The mother of all trading strategies.'''
    name         = None # string for later distinction
    symbols      = None # list of symbol strings matching alpha vantage data
    weights      = None # weights for all symbols, should add up to 1.0
    cash_buffer  = None # ratio of 
    frequency    = None # pd.Timedelta(), see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    data_span    = None # pd.Timedelta()

    def __init__(self):
        # Sanity checks and some asserts to test i.e. sum(weights) == 1
        pass
        
    @abc.abstractmethod
    def request(self, timestamp, data, portfolio, cash) -> List[Transaction]:
        '''Returns list of MeeTransaction or empty list'''
        # Note: Need to ensure, that data is actually available for the chosen period
        # Simply do nothing.
        return []
