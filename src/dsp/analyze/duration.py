from signal_module import Signal
from ..registry import register_analyze


@register_analyze("duration")
def duration(signal: Signal) -> float:
    return signal.data.size / signal.sr if signal.sr else 0.0
