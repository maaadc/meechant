#! /usr/bin/env python3

from meechant.backtest import BackTest
from meechant.strategies import benchmarks

if __name__ == '__main__':
    bt1 = BackTest(benchmarks.BuyAndHold())
    bt2 = BackTest(benchmarks.StockBonds6040())