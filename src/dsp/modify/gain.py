from signal_module import Signal
from ..registry import register_modify


@register_modify("gain")
def gain(signal: Signal, amount: float) -> Signal:
    out = signal.clone()
    out.data *= amount
    return out


@register_modify("gain_db")
def gain_db(signal: Signal, amount_db: float) -> Signal:
    out = signal.clone()
    amount = 10 ** (amount_db / 20)
    out.data *= amount
    return out
