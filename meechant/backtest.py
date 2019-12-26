from .marketdata import market

import collections
import pandas as pd
import numpy as np


Transaction = collections.namedtuple(
    'Transaction', ['time', 'symbol', 'amount', 'price'])


class BackTest():
    '''Simulation of a trading strategy with historical market data'''

    initial_cash = 1000.0
    broker_model = None
    cost_model = None
    start_date = pd.Timestamp('2004-01-01T20:00')
    end_date = pd.Timestamp('2019-12-01T20:00')

    _cash = 0.0
    _orders = []
    _stock_data = {}
    _strategy = None

    def __init__(self, trading_strategy):
        # Initialize private members
        self._cash = self.initial_cash
        self._orders = []
        self._stock_data = {}
        self._strategy = trading_strategy
        # Todo: Add sanity checks to ensure consistency of strategy

        # Gather necessary stock data
        for symbol in self._strategy.symbols:
            if symbol not in self._stock_data.keys():
                self._stock_data.update(
                    {symbol: market.get_daily(symbol)}
                )
        # Todo: Ensure that market data dates back long enough to cover the
        # start date and desired period for the strategy
        print(f'> Backtesting "{self._strategy.name}"')
        self.play()

    def play(self):
        '''Execute all of this'''
        # Execute all strategies for specified dates
        if self._strategy.frequency is None:
            date_list = [self.start_date]
        else:
            date_list = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq=self._strategy.frequency,
                closed=None)
        # Initialize variables. Note this binds cash currency to market currency
        self._portfolio = {symbol: 0.0 for symbol in self._strategy.symbols}
        self._cash = self.initial_cash
        # Trigger strategy at specified time intervals
        for timestamp in date_list:
            # Prepare stock market data
            stock_data = {
                symbol: self._stock_data[symbol].loc[(
                    timestamp-self._strategy.data_span):timestamp]
                for symbol in self._strategy.symbols}
            # Get requested orders from strategy
            new_orders = self._strategy.request(
                timestamp, stock_data, self._portfolio, self._cash)
            # Note: Risk analysis and cost model necessary at this point to determine if order gets executed
            # Note: Moar
            # Execute set of orders for next opening day
            if len(new_orders) > 0:
                for order in new_orders:
                    # Assume we execute the order for the opening price of the next day
                    execution_data = self._stock_data[order.symbol].loc[timestamp:].iloc[0]
                    execution_price = execution_data['open']
                    # Simulate the actual order, that would be executed with the broker
                    # Todo: Include
                    execution_cash = order.amount * order.price
                    execution_amount = execution_cash / execution_price
                    execution_order = Transaction(
                        time=execution_data.name,
                        symbol=order.symbol,
                        amount=execution_amount,
                        price=execution_price)
                    # Update internal parameters
                    self._portfolio[order.symbol] += execution_amount
                    self._cash -= execution_cash
                    self._orders.append(execution_order)
        # Clean up.
        print(f'''  {len(self._orders)} orders placed.''')

    def get(self):
        '''Return pd.Dataframe for all all '''
        # print(self._orders)
