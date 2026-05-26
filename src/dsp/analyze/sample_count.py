from signal_module import Signal
from ..registry import register_analyze


@register_analyze("sample_count")
def sample_count(signal: Signal) -> int:
    return int(signal.data.size)
