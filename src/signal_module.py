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

    def spectrogram(self, window_size: int = 512, hop_size: int | None = None, n_fft: int | None = None):
        """Compute a spectrogram (dB) using a windowed STFT.

        Returns (spec_db, freqs, times) where spec_db is shape (freq_bins, time_frames).
        """
        if hop_size is None:
            hop_size = max(1, window_size // 4)
        if n_fft is None:
            n_fft = window_size

        if self.data.size == 0 or self.sr == 0:
            return np.empty((0, 0)), np.array([]), np.array([])

        from dsp.analyze.spectrogram import spectrogram as analyze_spectrogram

        frames = analyze_spectrogram(self, window_size=window_size, hop_size=hop_size, n_fft=n_fft)
        if not frames:
            return np.empty((0, 0)), np.array([]), np.array([])

        spec = np.array(frames).T
        spec_db = 20 * np.log10(spec + 1e-9)
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sr)
        times = np.arange(len(frames)) * hop_size / self.sr
        return spec_db, freqs, times
