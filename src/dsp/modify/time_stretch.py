import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("time_stretch")
def time_stretch(signal: Signal, rate: float) -> Signal:
    if signal is None or signal.data.size == 0 or signal.sr == 0 or rate <= 0:
        return signal.clone()

    window_size = 1024
    hop = window_size // 4
    new_hop = max(1, int(round(hop / rate)))
    window = np.hanning(window_size)

    src = signal.data.astype(np.float64)
    frames = []
    for start in range(0, len(src) - window_size + 1, hop):
        frames.append(src[start : start + window_size] * window)

    if not frames:
        return signal.clone()

    out_length = max(len(src), new_hop * len(frames) + window_size)
    out = np.zeros(out_length, dtype=np.float64)
    pos = 0
    for frame in frames:
        out[pos : pos + window_size] += frame
        pos += new_hop

    if np.max(np.abs(out)) > 0:
        out *= np.max(np.abs(src)) / np.max(np.abs(out))
    return Signal(out, signal.sr)
