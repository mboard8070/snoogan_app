# trader/__init__.py - CORRECTED PACKAGE EXPORTS

# REMOVE: from trader.data_client import data_client (It's not in the 'trader' directory)
# Use relative imports for siblings (executor, indicators, etc.)
from .indicators import get_trend_signal
from .executor import Executor
from .position_manager import PositionManager
from .strategy import TradingStrategy

__all__ = [
    # "data_client",  # REMOVED
    "get_trend_signal",
    "Executor",
    "PositionManager",
    "TradingStrategy"
]