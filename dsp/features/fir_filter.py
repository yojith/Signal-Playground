import numpy as np

from signal_module import Signal
from ..registry import register


@register("fir_filter")
def fir_filter(signal: Signal, coefficients: list[float]) -> Signal:
    out = signal.clone()
    out.data = np.convolve(out.data, coefficients, mode="same").astype(np.float64)
    return out
