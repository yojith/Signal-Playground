import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("normalize")
def normalize(signal: Signal, target_peak: float = 1.0) -> Signal:
    if signal is None or signal.data.size == 0:
        return signal.clone()

    peak = np.max(np.abs(signal.data))
    if peak == 0:
        return signal.clone()

    scale = float(target_peak) / peak
    return Signal(signal.data * scale, signal.sr)
