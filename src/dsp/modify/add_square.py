import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("add_square")
def add_square(signal: Signal, freqs: list[float], amps: list[float]) -> Signal:
    out = signal.clone()
    t = np.arange(len(out.data)) / out.sr

    for freq, amp in zip(freqs, amps):
        out.data += amp * np.sign(np.sin(2 * np.pi * freq * t))
    return out
