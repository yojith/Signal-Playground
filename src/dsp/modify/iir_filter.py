import numpy as np
from scipy.signal import lfilter

from signal_module import Signal
from ..registry import register_modify


@register_modify("iir_filter")
def iir_filter(signal: Signal, b_coeffs: list[float], a_coeffs: list[float]) -> Signal:
    if signal is None or signal.data.size == 0 or signal.sr == 0:
        return signal.clone()

    b = np.asarray(b_coeffs, dtype=float)
    a = np.asarray(a_coeffs, dtype=float)
    if a.size == 0 or a[0] == 0:
        return signal.clone()

    y = lfilter(b, a, signal.data)
    return Signal(y, signal.sr)
