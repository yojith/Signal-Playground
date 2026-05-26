import numpy as np

from signal_module import Signal
from ..registry import register_analyze


@register_analyze("envelope")
def envelope(signal: Signal, method: str = "moving_average", window_size: int = 512) -> list[float]:
    if signal is None or signal.data.size == 0:
        return []

    data = np.abs(signal.data.astype(np.float64))
    if method == "moving_average":
        if window_size <= 1:
            return data.tolist()
        kernel = np.ones(window_size) / window_size
        envelope_data = np.convolve(data, kernel, mode="same")
        return envelope_data.tolist()

    return data.tolist()
