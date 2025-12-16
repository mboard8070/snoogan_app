# trader/__init__.py - package exports

from .data_client import data_client
from .indicators import get_trend_signal
from .executor import Executor
from .position_manager import PositionManager
from .strategy import TradingStrategy

__all__ = [
    "data_client",
    "get_trend_signal",
    "Executor",
    "PositionManager",
    "TradingStrategy"
]