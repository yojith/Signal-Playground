import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("limiter")
def limiter(signal: Signal, threshold_db: float) -> Signal:
    if signal is None or signal.data.size == 0:
        return signal.clone()

    threshold_lin = 10 ** (threshold_db / 20.0)
    clipped = np.clip(signal.data, -threshold_lin, threshold_lin)
    return Signal(clipped, signal.sr)
