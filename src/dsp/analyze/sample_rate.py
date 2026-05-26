from signal_module import Signal
from ..registry import register_analyze


@register_analyze("sample_rate")
def sample_rate(signal: Signal) -> int:
    return int(signal.sr)
