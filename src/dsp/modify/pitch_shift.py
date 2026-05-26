import librosa

from signal_module import Signal
from ..registry import register_modify


@register_modify("pitch_shift")
def pitch_shift(signal: Signal, semitones: int) -> Signal:
    out = signal.clone()
    out.data = librosa.effects.pitch_shift(out.data, sr=out.sr, n_steps=semitones)
    return out
