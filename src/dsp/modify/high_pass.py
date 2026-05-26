import numpy as np

from signal_module import Signal
from ..registry import register_modify
from .low_pass import low_pass_kernel


@register_modify("high_pass")
def high_pass(signal: Signal, cutoff_hz: float, num_taps: int = 101) -> Signal:
    if signal is None or signal.data.size == 0 or signal.sr == 0:
        return signal.clone()

    lowpass_coeffs = low_pass_kernel(cutoff_hz, signal.sr, num_taps)
    # Spectral inversion: delta[n] - h[n]
    impulse = np.zeros_like(lowpass_coeffs)
    center = (len(lowpass_coeffs) - 1) // 2
    impulse[center] = 1.0  # Impulse at the center
    highpass_taps = impulse - lowpass_coeffs

    filtered = np.convolve(signal.data, highpass_taps, mode="same")
    return Signal(filtered, signal.sr)
