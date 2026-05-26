import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("delay")
def delay(signal: Signal, delay_seconds: float) -> Signal:
    if signal is None or signal.data.size == 0 or signal.sr == 0:
        return signal.clone()

    delay_samples = int(round(delay_seconds * signal.sr))
    if delay_samples <= 0:
        return signal.clone()

    delayed = np.concatenate([np.zeros(delay_samples, dtype=signal.data.dtype), signal.data])
    return Signal(delayed, signal.sr)
