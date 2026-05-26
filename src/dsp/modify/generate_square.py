import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("generate_square")
def generate_square(signal: Signal, frequencies: list[float], duration: float, sr: int) -> Signal:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = np.zeros_like(t)
    for freq in frequencies:
        data += np.sign(np.sin(2 * np.pi * freq * t))

    data /= len(frequencies)
    return Signal(data, sr)
