#! /usr/bin/env python3

from meechant.backtest import BackTest
from meechant.strategies import benchmarks

if __name__ == '__main__':
    bt1 = BackTest(benchmarks.KeepTheCash())
    print(bt1.timeline.tail())
    bt2 = BackTest(benchmarks.BuyAndHold())
    print(bt2.timeline.tail())
    bt3 = BackTest(benchmarks.StockBonds6040())
    print(bt3.timeline.tail())
