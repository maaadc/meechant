#! /usr/bin/env python3

from meechant.backtest import BackTest
from meechant.strategies import benchmarks

if __name__ == '__main__':
    bt1 = BackTest(benchmarks.StatusQuo()).get()
    bt2 = BackTest(benchmarks.BuyAndHold()).get()
    bt3 = BackTest(benchmarks.StockBonds6040()).get()
