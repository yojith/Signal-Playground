import numpy as np

from signal_module import Signal
from ..registry import register_analyze


@register_analyze("zero_crossing_rate")
def zero_crossing_rate(signal: Signal) -> float:
    if signal is None or signal.data.size == 0:
        return 0.0
    data = signal.data.astype(np.float64)
    crossings = np.sum(np.abs(np.diff(np.sign(data)))) / 2.0
    return float(crossings / max(len(data) - 1, 1))
