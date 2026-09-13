"""
Analytics layer for QuantSim.

Re-exports the public pieces so calling code can do:

    from src.analytics import PerformanceMetrics, calculate_performance_metrics

instead of reaching into performance_metrics.py directly.
"""

from src.analytics.performance_metrics import (
    PerformanceMetrics,
    calculate_performance_metrics,
)

__all__ = [
    "PerformanceMetrics",
    "calculate_performance_metrics",
]
