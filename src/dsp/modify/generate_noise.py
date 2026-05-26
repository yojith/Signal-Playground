import numpy as np

from signal_module import Signal
from ..registry import register_modify


@register_modify("generate_noise")
def generate_noise(signal: Signal, std: float) -> Signal:
    out = signal.clone()
    rng = np.random.default_rng()
    noise = rng.normal(0, std, size=len(out.data))  # Generate white Gaussian noise
    out.data += noise
    return out
