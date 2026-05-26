import numpy as np

from signal_module import Signal
from ..registry import register_modify
from .low_pass import low_pass_kernel


@register_modify("band_pass")
def band_pass(signal: Signal, low_cutoff_hz: float, high_cutoff_hz: float, num_taps: int = 101) -> Signal:
    if signal is None or signal.data.size == 0 or signal.sr == 0:
        return signal.clone()
    if low_cutoff_hz >= high_cutoff_hz:
        return signal.clone()

    high_kernel = low_pass_kernel(high_cutoff_hz, signal.sr, num_taps)
    low_kernel = low_pass_kernel(low_cutoff_hz, signal.sr, num_taps)
    band_kernel = high_kernel - low_kernel
    filtered = np.convolve(signal.data, band_kernel, mode="same")
    return Signal(filtered, signal.sr)
