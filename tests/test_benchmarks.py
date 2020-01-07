from meechant.backtester import BackTester
from meechant.strategies import benchmarks


def test_benchmark_execution():
    # Execute all benchmark strategies once, omit testing functionality for now
    BackTester(benchmarks.KeepTheCash())
    BackTester(benchmarks.BuyAndHold())
    BackTester(benchmarks.StockBonds6040())
    BackTester(benchmarks.MovingAverageCrossover())
