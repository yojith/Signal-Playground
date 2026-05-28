import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("truncate")
def truncate(signal: Signal, start_time: float | None = None, end_time: float | None = None, step: float | None = None) -> Signal:
    if start_time is None and end_time is None and step is None:
        return signal.clone()

    if start_time is None:
        start_sample = 0
    else:
        start_sample = int(start_time * signal.sr)

    if end_time is None:
        end_sample = signal.data.size
    else:
        end_sample = int(end_time * signal.sr)

    np.clip(start_sample, 0, signal.data.size)
    np.clip(end_sample, 0, signal.data.size)

    if start_sample >= end_sample:
        return Signal.empty(sr=signal.sr)

    truncated_data = signal.data[start_sample:end_sample]
    return Signal(data=truncated_data, sr=signal.sr)
