
import seaborn as sns
from typing import Dict
from .marketdata import market
from .strategies import Transaction
import numpy as np
import pandas as pd
# plotting
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set()
sns.set_context('talk')  # enlarged font
plt.rc('figure', figsize=(16, 10))  # default figure size


class BackTester():
    '''Simulation of a trading strategy with historical market data'''
    # Parameters for the actual backtesting
    initial_cash = 1000.0
    broker_model = None
    cost_model = None
    start_date = pd.Timestamp('2005-01-01T20:00')
    end_date = pd.Timestamp('2020-01-01T20:00')
    # Parameters for the statistical analysis
    symbol_risk_free = '^FVX'  # Treasury Yield 5 Years
    symbol_market = '^GSPC'  # The asset used for comparison with the market, here we use the the S&P 500
    # output variables
    name = ''
    stats = {}
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
        for symbol in self._strategy.symbols + [self.symbol_risk_free, self.symbol_market]:
            if symbol not in self._stock_data.keys():
                self._stock_data.update(
                    {symbol: market.get_daily(symbol)}
                )
        # Todo: Ensure that market data dates back long enough to cover the
        # start date and desired period for the strategy
        print(
            f'> Backtesting \x1b[93m{self._strategy.name}\x1b[0m from {self.start_date} to {self.end_date}')
        self.run()

    def _calc_statistics(self) -> Dict[str, float]:
        # Note: All return rates are given on an annual basis, if not stated otherwise.
        stats = {'name': self.name}
        # Compute ratio of change between two consecutive data points in timeline.
        symbol_returns = self.timeline.pct_change(periods=1).dropna()
        # Get total return rate over the whole sampling period
        stats['total'] = symbol_returns.agg(lambda x: (x + 1).prod() - 1)
        # Get averaged annual return rate
        duration_years = (self.end_date - self.start_date).days / 365
        stats['annual'] = (1.0 + stats['total'])**(1.0 / duration_years) - 1.0
        # Define annual risk free rate and compute it for each business day.
        # Todo: Derive from some kind of index
        risk_free_rate = 0.015
        risk_free_bdaily = (1.0 + risk_free_rate)**(1.0 / 252) - 1.0
        # Get market returns serving as a benchmark
        market_returns = self._stock_data[self.symbol_market][self.start_date:self.end_date][
            'close'].pct_change(periods=1).dropna()
        # Compute alpha and beta coefficients of the capital asset pricing model (CAPM).
        # See https://en.wikipedia.org/wiki/Alpha_(finance) and https://en.wikipedia.org/wiki/Beta_(finance)
        r_market = market_returns - risk_free_bdaily
        r_symbol = symbol_returns - risk_free_bdaily
        stats['beta'] = r_symbol.cov(r_market) / r_market.var()
        stats['alpha'] = r_symbol.mean() - stats['beta'] * r_market.mean()
        # Compute averaged indicators based on daily returns.
        if r_symbol.std() > 0:
            # Sharpe ratio (annual)
            # see https://en.wikipedia.org/wiki/Sharpe_ratio
            stats['Sharpe'] = np.sqrt(252) * r_symbol.mean() / r_symbol.std()
            # Modigliani risk-adjusted performance (annual)
            # see https://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance
            stats['M2'] = 252 * r_symbol.mean() * r_market.std() / r_symbol.std() + risk_free_rate
        return stats

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
            # Note: Buy as much as possible with the invested cash and sell everything we own
            if order.amount > 0:
                execution_cash = order.amount * order.price
                execution_amount = execution_cash / execution_price
            else:
                # Sell everything
                execution_amount = order.amount
                execution_cash = order.amount * execution_price
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
        self.stats = self._calc_statistics()
        # Show some stats and clean up.
        print(
            f'  Annual return is \x1b[1m{100*self.stats["annual"]:5.2f} %\x1b[0m with {len(self._orders)} orders')

    def plot(self) -> matplotlib.figure.Figure:
        '''Creates an overview plot of the backtesting performance'''
        num_symbols = len(self._strategy.symbols)
        # create figure and axes
        fig, ax = plt.subplots(nrows=num_symbols+1, ncols=1, sharex=True,
                               figsize=(16, 4*(num_symbols+1)))
        fig.suptitle(self.name, fontsize=24)
        # plot stock data with annotated order symbols
        for k in range(len(self._strategy.symbols)):
            symbol = self._strategy.symbols[k]
            # get stock data for the specified time frame
            df = market.get_daily(symbol)
            df = df[self.start_date:self.end_date]
            # plot closing price for specified symbol and derived exponential moving averages
            ax[k].plot(df['close'], 'k-', color=3*[0.25])
            ax[k].plot(df['close'].ewm(halflife=30, adjust=False).mean(), 'b-.')
            ax[k].plot(df['close'].ewm(halflife=150, adjust=False).mean(), 'b--')
            # plot buying orders
            amount = np.array(
                [o.amount for o in self._orders if o.amount > 0 and symbol == o.symbol])
            ax[k].scatter([o.time for o in self._orders if o.amount > 0 and symbol == o.symbol],
                          [o.price for o in self._orders if o.amount > 0 and symbol == o.symbol],
                          s=20**2 * np.sqrt(amount/amount.mean()),
                          c='green', marker='^', alpha=0.7, zorder=10)
            # plot selling orders
            amount = np.array([np.abs(o.amount)
                               for o in self._orders if o.amount < 0 and symbol == o.symbol])
            ax[k].scatter([o.time for o in self._orders if o.amount < 0 and symbol == o.symbol],
                          [o.price for o in self._orders if o.amount < 0 and symbol == o.symbol],
                          s=20**2 * np.sqrt(amount/amount.mean()),
                          c='red', marker='v', alpha=0.7, zorder=10)
            # beautify
            ax[k].set_ylabel(f'{symbol} (USD)')
        # add legend to first plot
        ax[0].legend(['symbol', 'EMA30', 'EMA150', 'buy', 'sell'],
                     loc='best', scatterpoints=1, markerscale=0.8, ncol=5)
        # plot total wealth over time
        ax[-1].plot(self.timeline, 'k-', color=3*[0.25])
        ax[-1].set_ylabel('wealth (USD)')
        ax[-1].set_xlabel('time (years)')
        # add statistics output
        stats_str = (f'Returns: {100*self.stats["total"]:3.2f} % total, {100*self.stats["annual"]:3.2f} % annual\n'
                     f'CAPM: alpha = {self.stats["alpha"]:1.4f}, beta = {self.stats["beta"]:1.4f}\n'
                     f'Sharpe = {self.stats["Sharpe"]:.3f}, M2 = {100*self.stats["M2"]:.3f} %')
        ax[-1].text(0.01, 0.95, stats_str, transform=ax[-1].transAxes,
                    verticalalignment='top')
        return fig
