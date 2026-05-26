import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("harmonic_distortion")
def harmonic_distortion(signal: Signal, drive: float = 1.0) -> Signal:
    if signal is None or signal.data.size == 0:
        return signal.clone()

    x = signal.data.astype(np.float64)
    drive = float(drive)
    if drive == 0:
        return signal.clone()

    shaped = np.tanh(drive * x)
    shaped /= np.tanh(drive)
    return Signal(shaped, signal.sr)
