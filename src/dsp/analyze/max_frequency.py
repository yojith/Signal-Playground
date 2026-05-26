import numpy as np

from signal_module import Signal
from ..registry import register_analyze


@register_analyze("max_frequency")
def max_frequency(signal: Signal) -> float | None:
    if signal.data.size == 0 or signal.sr == 0:
        return None

    spectrum = np.abs(np.fft.rfft(signal.data))
    freqs = np.fft.rfftfreq(signal.data.size, d=1.0 / signal.sr)
    peak_index = int(np.argmax(spectrum))
    return float(freqs[peak_index])
