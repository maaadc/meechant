from .marketdata import market

import collections
import pandas as pd
import numpy as np

from typing import Dict


Transaction = collections.namedtuple('Transaction', ['time', 'symbol', 'amount', 'price'])


class BackTest():
    '''Simulation of a trading strategy with historical market data'''
    # input variables
    initial_cash = 1000.0
    broker_model = None
    cost_model = None
    start_date = pd.Timestamp('2004-12-01T20:00')
    end_date = pd.Timestamp('2019-12-01T20:00')
    # output variables
    name = ''
    timeline = pd.DataFrame({})

    # private members
    _cash = 0.0
    _orders = []
    _stock_data = {}
    _strategy = None

    def __init__(self, trading_strategy) -> None:
        # Initialize private members
        self._orders = []
        self._stock_data = {}
        self._strategy = trading_strategy
        self.name = self._strategy.name
        # Todo: Add sanity checks to ensure consistency of strategy
        # Gather necessary stock data
        for symbol in self._strategy.symbols:
            if symbol not in self._stock_data.keys():
                self._stock_data.update(
                    {symbol: market.get_daily(symbol)}
                )
        # Todo: Ensure that market data dates back long enough to cover the
        # start date and desired period for the strategy
        self.run()

    def _calc_timeline(self) -> pd.Series:
        '''Return pd.Dataframe for all all ... only where stock data is available.'''
        # Handle the trivial case without any stock symbols or no orders at all
        if len(self._stock_data.keys()) == 0 or len(self._orders) == 0:
            # Return the initial cash for a set of arbitrary dates: 'SMS' is 1st and 15th of each month
            date_list = pd.date_range(
                start=self.start_date, end=self.end_date, freq='SMS', closed=None)
            return pd.Series(self.initial_cash, index=date_list)
        # Get closing prices for all stock symbols in the specified time span
        dfn = pd.concat([
            self._stock_data[symbol][self.start_date:self.end_date]['close'].rename(symbol)
            for symbol in self._strategy.symbols], axis=1, join='inner')
        # Add a column to track the total wealth including cash
        dfn['total'] = self.initial_cash
        # Since all input data is available we can apply the transactions sequentially
        # Note: Orders are executed for the opening price and wealth is estimated for the closing price
        for o in self._orders:
            # Starting from the time of transaction, we need to do two things:
            # Subtract the amount of cash that has been paid for the order and
            # add the up-to-date market price this corresponds to
            dfn['total'][o.time:] += o.amount * (dfn[o.symbol][o.time:] - o.price)
        # Finished. Only return the total.
        return dfn['total']

    def _execute_orders(self, requested_orders) -> None:
        '''Execute transactions that have been requested by a strategy'''
        for order in requested_orders:
            # Assume we execute the order for the opening price of the next day
            execution_data = self._stock_data[order.symbol].loc[order.time:].iloc[0]
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

    def run(self) -> None:
        '''Execute all of this'''
        # Initialize variables. Note: Binds cash currency to market currency
        self._portfolio = {symbol: 0.0 for symbol in self._strategy.symbols}
        self._cash = self.initial_cash
        # Execute all strategies for specified dates
        if self._strategy.frequency is None:
            date_list = [self.start_date]
        else:
            date_list = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq=self._strategy.frequency,
                closed=None)
        # Trigger strategy at specified time intervals
        for timestamp in date_list:
            # Prepare stock market data for the requested time interval
            stock_data = {
                symbol: self._stock_data[symbol].loc[(
                    timestamp-self._strategy.data_span):timestamp]
                for symbol in self._strategy.symbols}
            # Get requested orders from strategy
            new_orders = self._strategy.request(
                timestamp, stock_data, self._portfolio, self._cash)
            # Note: Risk analysis and cost model necessary at this point to determine if order gets executed
            # This is crucial.
            # Execute set of orders for next opening day
            self._execute_orders(new_orders)
        # Determine wealth over time for the executed transactions
        self.timeline = self._calc_timeline()
        # Show some stats and Clean up.
        print(f'> Backtest "{self._strategy.name}" with {len(self._orders)} orders placed.''')
