import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("echo")
def echo(signal: Signal, delay_seconds: float, decay: float) -> Signal:
    if signal is None or signal.data.size == 0 or signal.sr == 0:
        return signal.clone()

    delay_samples = int(round(delay_seconds * signal.sr))
    if delay_samples <= 0 or decay == 0:
        return signal.clone()

    out = signal.data.copy().astype(np.float64)
    max_repeats = 3
    for i in range(1, max_repeats + 1):
        start = delay_samples * i
        if start >= len(out) + delay_samples:
            break
        echo_signal = np.pad(signal.data * (decay**i), (start, 0), mode="constant")
        if echo_signal.size > out.size:
            out = np.pad(out, (0, echo_signal.size - out.size), mode="constant")
        out[: echo_signal.size] += echo_signal
    return Signal(out, signal.sr)
