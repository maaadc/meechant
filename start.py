#! /usr/bin/env python3

from meechant.backtest import BackTest
from meechant.strategies import benchmarks

if __name__ == '__main__':
    bt1 = BackTest(benchmarks.KeepTheCash())

    bt2 = BackTest(benchmarks.BuyAndHold())

    bt3 = BackTest(benchmarks.StockBonds6040())

    bt4 = BackTest(benchmarks.MovingAverageCrossover())
    # print(bt4.timeline)
    print(bt4.stats)
    # print(bt4._orders)
