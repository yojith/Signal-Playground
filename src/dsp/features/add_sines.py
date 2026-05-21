import numpy as np

from signal_module import Signal
from ..registry import register


@register("add_sines")
def add_sines(signal: Signal, freqs: list[float], amps: list[float]) -> Signal:
    out = signal.clone()
    t = np.arange(len(out.data)) / out.sr  # Time axis by dividing number of samples by sample rate

    for freq, amp in zip(freqs, amps):
        out.data += amp * np.sin(2 * np.pi * freq * t)
    return out
