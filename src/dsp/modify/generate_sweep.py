import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("generate_sweep")
def generate_sweep(
    signal: Signal,
    start_freq: float,
    end_freq: float,
    duration: float,
    sr: int,
    sweep_type: str = "linear",
) -> Signal:
    sample_count = int(sr * duration)
    if sample_count <= 0 or duration <= 0:
        return Signal(np.array([], dtype=np.float64), sr)

    t = np.linspace(0, duration, sample_count, endpoint=False)
    start_freq = float(start_freq)
    end_freq = float(end_freq)

    if sweep_type.lower() in {"log", "logarithmic"} and start_freq > 0 and end_freq > 0:
        if start_freq == end_freq:
            phase = 2 * np.pi * start_freq * t
        else:
            k = np.log(end_freq / start_freq) / duration
            phase = 2 * np.pi * start_freq * (np.exp(k * t) - 1) / k
    else:
        sweep_rate = (end_freq - start_freq) / duration
        phase = 2 * np.pi * (start_freq * t + 0.5 * sweep_rate * t**2)

    data = np.sin(phase)

    fade_time = min(0.02, duration * 0.05)
    fade_samples = max(1, int(sr * fade_time))
    if fade_samples * 2 < sample_count:
        envelope = np.ones(sample_count, dtype=np.float64)
        fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
        fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=False)
        envelope[:fade_samples] = fade_in
        envelope[-fade_samples:] = fade_out
        data *= envelope

    return Signal(data, sr)
