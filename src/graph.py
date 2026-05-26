import numpy as np
import plotly.graph_objects as go

from signal_module import Signal


class SignalGraph:
    def __init__(self, signal: Signal) -> None:
        self.signal = signal

    def plot_time_domain(self, st) -> None:
        data = self.signal.data  # Time-domain data
        sr = self.signal.sr
        t = np.arange(len(data)) / sr  # Time axis in seconds

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=data, mode="lines", name="Signal"))
        fig.update_layout(title="Time Domain", xaxis_title="Time (s)", yaxis_title="Amplitude", height=350)
        st.plotly_chart(fig, width="stretch")

    def plot_frequency_domain(self, st) -> None:
        data = self.signal.data
        sr = self.signal.sr

        fft = np.fft.rfft(data)  # Data in frequency domain (Only real part)
        freq = np.fft.rfftfreq(len(data), 1 / sr)  # Frequency axis in Hz
        magnitude = np.abs(fft)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freq, y=magnitude, mode="lines", name="FFT"))
        fig.update_layout(title="Frequency Domain", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", height=350)
        st.plotly_chart(fig, width="stretch")

    def plot_spectrogram(self, st, window_size: int = 512, hop_size: int | None = None, n_fft: int | None = None) -> None:
        spec, freqs, times = self.signal.spectrogram(window_size=window_size, hop_size=hop_size, n_fft=n_fft)
        if spec.size == 0:
            return

        fig = go.Figure(
            data=go.Heatmap(
                z=spec,
                x=times,
                y=freqs,
                colorscale="Viridis",
                colorbar=dict(title="dB"),
            )
        )
        fig.update_layout(
            title="Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            height=350,
        )
        st.plotly_chart(fig, width="stretch")
