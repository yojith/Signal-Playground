import numpy as np

from signal_module import Signal
from ..registry import register_modify


def low_pass_kernel(cutoff_hz: float, sr: int, size: int = 101) -> np.ndarray:
    fc_norm = cutoff_hz / sr
    n = np.arange(size) - (size - 1) / 2  # Eg. [0, 1, 2, 3, 4] becomes [-2, -1, 0, 1, 2]
    h = 2 * fc_norm * np.sinc(2 * fc_norm * n)
    window = np.hamming(size)
    h *= window
    h /= np.sum(h)  # All elements must add to 1
    return h


@register_modify("low_pass")
def low_pass(signal: Signal, cutoff_hz: float, num_taps: int = 101) -> Signal:
    if signal is None or signal.data.size == 0 or signal.sr == 0:
        return signal.clone()

    kernel = low_pass_kernel(cutoff_hz, signal.sr, num_taps)
    filtered = np.convolve(signal.data, kernel, mode="same")
    return Signal(filtered, signal.sr)
