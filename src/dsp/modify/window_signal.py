import numpy as np

from signal_module import Signal
from ..registry import register_modify


def _window(name: str, size: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(size)
    if name == "hamming":
        return np.hamming(size)
    if name == "blackman":
        return np.blackman(size)
    if name == "bartlett":
        return np.bartlett(size)
    return np.ones(size)


@register_modify("window_signal")
def window_signal(signal: Signal, window_type: str = "hann", size: int | None = None) -> Signal:
    if signal is None or signal.data.size == 0:
        return signal.clone()

    length = len(signal.data)
    if size is None or size <= 0 or size > length:
        size = length

    window = _window(window_type.lower(), size)
    envelope = np.ones(length)
    envelope[:size] = window
    if size < length:
        envelope[-size:] = window
    return Signal(signal.data * envelope, signal.sr)
