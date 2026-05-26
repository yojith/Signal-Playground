import numpy as np

from signal_module import Signal
from ..registry import register_modify


def _smooth_envelope(level: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    attack = np.exp(-1.0 / max(sr * attack_ms * 0.001, 1e-9))
    release = np.exp(-1.0 / max(sr * release_ms * 0.001, 1e-9))
    env = np.zeros_like(level)
    for i, value in enumerate(level):
        coeff = attack if value > env[i - 1] else release
        env[i] = coeff * env[i - 1] + (1 - coeff) * value if i > 0 else value
    return env


@register_modify("compressor")
def compressor(
    signal: Signal,
    threshold_db: float,
    ratio: float,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
) -> Signal:
    if signal is None or signal.data.size == 0 or signal.sr == 0:
        return signal.clone()

    threshold_lin = 10 ** (threshold_db / 20.0)
    x = signal.data.astype(np.float64)
    level = np.abs(x)
    smoothed = _smooth_envelope(level, signal.sr, attack_ms, release_ms)

    gain = np.ones_like(smoothed)
    over = smoothed > threshold_lin
    gain[over] = threshold_lin + (smoothed[over] - threshold_lin) / max(ratio, 1.0)
    gain = gain / np.maximum(smoothed, 1e-9)

    return Signal(x * gain, signal.sr)
