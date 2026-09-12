"""
Strategy layer for QuantSim.

Re-exports the common pieces so calling code can do:

    from src.strategies import Signal, MovingAverageCrossoverStrategy, RSIStrategy, MACDStrategy

instead of reaching into each individual file.
"""

from src.strategies.base_strategy import BaseStrategy, Signal, StrategyInputError
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.moving_average_strategy import MovingAverageCrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "StrategyInputError",
    "MovingAverageCrossoverStrategy",
    "RSIStrategy",
    "MACDStrategy",
]
