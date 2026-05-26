import numpy as np

from signal_module import Signal
from ..registry import register_analyze


@register_analyze("rms_level")
def rms_level(signal: Signal) -> float:
    if signal is None or signal.data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal.data.astype(np.float64) ** 2)))
