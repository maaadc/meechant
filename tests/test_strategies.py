from meechant.strategies import TradingStrategy, Transaction

import pandas as pd


def test_transaction():
    # Come up with example values
    timestamp = pd.Timestamp.now()
    symbol = 'SPY'
    amount = 12.34
    price = 56.78
    # Create transaction and test its keys
    # Note, that this also tests the order of arguments
    t = Transaction(timestamp, symbol, amount, price)
    assert t.time == timestamp
    assert t.symbol == symbol
    assert t.amount == amount
    assert t.price == price
