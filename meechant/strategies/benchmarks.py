from . import TradingStrategy
from ..backtest import Transaction

import pandas as pd
import numpy as np


class KeepTheCash(TradingStrategy):
    '''Strategy which does exactly nothing.'''
    name = 'KeepTheCash'
    symbols = []
    weights = []
    cash_buffer = 0.0
    frequency = None
    data_span = pd.Timedelta('1W')

    def request(self, timestamp, data, portfolio, cash):
        return []


class BuyAndHold(TradingStrategy):
    '''Strategy to buy and hold an index tracking the S&P 500.'''
    name = 'BuyAndHold SPY'
    symbols = ['SPY']
    weights = [1.0]
    cash_buffer = 0.0
    frequency = None
    data_span = pd.Timedelta('1W')

    def request(self, timestamp, data, portfolio, cash):
        # Get latest closing price, which is likely the next price we can buy something for
        latest_price = data[self.symbols[0]]['close'].iloc[-1]
        # Only one transaction needed
        return [Transaction(
            time=timestamp,
            symbol=self.symbols[0],
            amount=cash / latest_price,
            price=latest_price
        )]


class StockBonds6040(TradingStrategy):
    '''Benchmark: Stock & bonds at a 60:40 ratio

    https://www.quantstart.com/articles/the-6040-benchmark-portfolio/

    # SPY - SPDR S&P500 ETF Trust, tracking the S&P500 US large-cap stock market index. Alternative ETFs include IVV and VOO.
    # AGG - iShares Core U.S. Aggregate Bond ETF, tracking an index of US investment-grade bonds. An alternative ETF is BND.
    '''
    name = '60:40 Stock&Bonds'
    symbols = ['SPY', 'AGG']
    weights = [0.60, 0.40]
    cash_buffer = 0.0
    frequency = pd.tseries.offsets.MonthBegin()
    data_span = pd.Timedelta('1W')

    def request(self, timestamp, data, portfolio, cash):
        # Get latest closing price, which is likely the next price we can buy something for
        orders = []
        # print(portfolio)
        # Calculate weights for current market data
        current_weights = [data[symbol]['close'].iloc[-1] * portfolio[symbol]
                           for symbol in self.symbols]
        if np.sum(current_weights) > 0:
            current_weights /= np.sum(current_weights)
        # Only suggest an order if weights drifted too far
        for symbol, target_weight, current_weight in zip(self.symbols, self.weights, current_weights):
            order_price = data[symbol]['close'].iloc[-1]
            total_cash = cash + np.sum([data[symbol]['close'].iloc[-1]
                                        * portfolio[symbol] for symbol in self.symbols])
            order_amount = (1.0 - self.cash_buffer) * total_cash * \
                (target_weight - current_weight) / order_price
            # print([symbol, current_weight, target_weight, np.abs(current_weight - target_weight)])
            if np.abs(current_weight - target_weight) > 0.01:
                orders.append(Transaction(timestamp, symbol, order_amount, order_price))
        return orders
