#! /usr/bin/env python3

from meechant.backtester import BackTester
from meechant.strategies import benchmarks

if __name__ == '__main__':
    bt1 = BackTester(benchmarks.KeepTheCash())

    bt2 = BackTester(benchmarks.BuyAndHold())

    bt3 = BackTester(benchmarks.StockBonds6040())

    bt4 = BackTester(benchmarks.MovingAverageCrossover())
    # print(bt4.timeline)
    print(bt4.stats)
    # print(bt4._orders)
