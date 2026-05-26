import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("clip")
def clip(signal: Signal, threshold: float) -> Signal:
    if signal is None or signal.data.size == 0:
        return signal.clone()

    clipped = np.clip(signal.data, -abs(threshold), abs(threshold))
    return Signal(clipped, signal.sr)
