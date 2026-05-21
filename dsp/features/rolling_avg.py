import numpy as np

from signal_module import Signal
from ..registry import register


@register("rolling_avg")
def rolling_avg(signal: Signal, window_size: int) -> Signal:
    out = signal.clone()
    filter_window = np.ones(window_size) / window_size
    out.data = np.convolve(out.data, filter_window, mode="same").astype(np.float64)
    return out
