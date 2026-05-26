import numpy as np

from signal_module import Signal
from ..registry import register_modify


def _all_pass_section(data: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(data)
    prev_x = 0.0
    prev_y = 0.0
    for n, x in enumerate(data):
        out[n] = alpha * x + prev_x - alpha * prev_y
        prev_x = x
        prev_y = out[n]
    return out


@register_modify("all_pass_filter")
def all_pass_filter(signal: Signal, phase_params: list[float]) -> Signal:
    if signal is None or signal.data.size == 0 or signal.sr == 0:
        return signal.clone()

    params = phase_params if isinstance(phase_params, list) else [phase_params]
    data = signal.data.copy()
    for alpha in params:
        data = _all_pass_section(data, float(alpha))
    return Signal(data, signal.sr)
