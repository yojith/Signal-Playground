import io
import librosa
import numpy as np
import soundfile as sf

from typing import BinaryIO


class Signal:
    def __init__(self, data: np.ndarray, sr: int) -> None:
        self.data = data.astype(np.float64)
        self.sr = sr

    @classmethod
    def empty(cls, sr: int = 16000) -> "Signal":
        return Signal(np.array([0]), sr)

    def is_empty(self) -> bool:
        return self.data.size == 1 and self.data[0] == 0

    @classmethod
    def from_wav(cls, file: BinaryIO | str) -> "Signal":
        data, sr = librosa.load(file, sr=16000, mono=True)
        return cls(data, int(sr))

    def clone(self) -> "Signal":
        return Signal(self.data.copy(), self.sr)

    def to_audio_bytes(self) -> io.BytesIO:
        buf = io.BytesIO()
        sf.write(buf, self.data, self.sr, format="WAV")
        buf.seek(0)
        return buf
