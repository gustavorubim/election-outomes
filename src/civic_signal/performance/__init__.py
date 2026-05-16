"""Performance helpers and accelerated kernels."""

from civic_signal.performance.kernels import (
    NUMBA_AVAILABLE,
    binary_draw_kernel,
    configure_numba_threads,
    simulate_binary_draw_arrays,
)

__all__ = [
    "NUMBA_AVAILABLE",
    "binary_draw_kernel",
    "configure_numba_threads",
    "simulate_binary_draw_arrays",
]
