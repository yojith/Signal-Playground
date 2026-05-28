import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("generate_step_sweep")
def generate_step_sweep(
    signal: Signal,
    start_freq: float,
    end_freq: float,
    num_intervals: int,
    interval_duration: float,
    sr: int,
    sweep_type: str = "linear",
) -> Signal:
    num_intervals = int(num_intervals)
    interval_duration = float(interval_duration)
    sample_count = int(sr * interval_duration * num_intervals)
    if sample_count <= 0 or num_intervals <= 0 or interval_duration <= 0:
        return Signal(np.array([], dtype=np.float64), sr)

    if num_intervals == 1:
        frequencies = np.array([float(start_freq)])
    elif sweep_type.lower() in {"log", "logarithmic"} and float(start_freq) > 0 and float(end_freq) > 0:
        frequencies = np.geomspace(float(start_freq), float(end_freq), num_intervals)
    else:
        frequencies = np.linspace(float(start_freq), float(end_freq), num_intervals)

    interval_samples = int(sr * interval_duration)
    if interval_samples <= 0:
        return Signal(np.array([], dtype=np.float64), sr)

    data = np.zeros(sample_count, dtype=np.float64)
    phase = 0.0
    for idx, freq in enumerate(frequencies):
        start = idx * interval_samples
        end = start + interval_samples
        if end > sample_count:
            end = sample_count
        segment_len = end - start
        if segment_len <= 0:
            break

        t = np.arange(segment_len, dtype=np.float64) / sr
        segment = np.sin(2 * np.pi * freq * t + phase)
        phase += 2 * np.pi * freq * segment_len / sr
        phase = np.mod(phase, 2 * np.pi)
        data[start:end] = segment

    fade_time = min(0.01, interval_duration * 0.25)
    fade_samples = max(1, int(sr * fade_time))
    if fade_samples * 2 < sample_count:
        envelope = np.ones(sample_count, dtype=np.float64)
        fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
        fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=False)
        envelope[:fade_samples] = fade_in
        envelope[-fade_samples:] = fade_out
        data *= envelope

    return Signal(data, sr)
