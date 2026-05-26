from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signal_module import Signal

from ..registry import register_analyze


@register_analyze("spectrogram")
def spectrogram(signal: Signal, window_size: int = 512, hop_size: int = 256, n_fft: int | None = None) -> list[list[float]]:
    if signal is None or signal.data.size == 0 or signal.sr == 0:
        return []

    if n_fft is None:
        n_fft = window_size
    window = np.hanning(window_size)
    data = signal.data.astype(np.float64)
    frames = []
    for start in range(0, len(data) - window_size + 1, hop_size):
        frame = data[start : start + window_size] * window
        spectrum = np.abs(np.fft.rfft(frame, n=n_fft))
        frames.append(spectrum.tolist())
    return frames
