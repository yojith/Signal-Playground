import numpy as np

from signal_module import Signal
from ..registry import register_analyze


@register_analyze("top_frequencies")
def top_frequencies(signal: Signal, top_n: int) -> list[dict[str, float]]:
    if signal.data.size == 0 or signal.sr == 0:
        return []

    spectrum = np.abs(np.fft.rfft(signal.data))
    freqs = np.fft.rfftfreq(signal.data.size, d=1.0 / signal.sr)
    indices = np.argsort(spectrum)[::-1][:top_n]
    return [
        {
            "frequency_hz": float(freqs[i]),
            "magnitude": float(spectrum[i]),
        }
        for i in indices
    ]
