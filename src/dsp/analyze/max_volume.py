import numpy as np

from signal_module import Signal
from ..registry import register_analyze


@register_analyze("max_volume")
def max_volume(signal: Signal) -> dict[str, float | None]:
    max_amp = float(np.max(np.abs(signal.data)))
    if max_amp == 0:
        return {"max_amplitude": 0.0, "dbfs": None}

    return {
        "max_amplitude": max_amp,
        "dbfs": float(20 * np.log10(max_amp)),
    }
