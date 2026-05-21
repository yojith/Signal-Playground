import numpy as np

from signal_module import Signal
from ..registry import register


@register("generate_sines_phase")
def generate_sines_phase(signal: Signal, frequencies: list[float], phases: list[float], duration: float, sr: int) -> Signal:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = np.zeros_like(t)
    for freq, phase in zip(frequencies, phases):
        data += np.sin(2 * np.pi * freq * t + phase)

    data /= len(frequencies)  # Normalize
    return Signal(data, sr)


@register("generate_sines")
def generate_sines(signal: Signal, frequencies: list[float], duration: float, sr: int) -> Signal:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = np.zeros_like(t)
    for freq in frequencies:
        data += np.sin(2 * np.pi * freq * t)

    data /= len(frequencies)  # Normalize
    return Signal(data, sr)
