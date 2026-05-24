import numpy as np

from signal_module import Signal
from ..registry import register

WIDTH = 64
DENSITY = 2048


def sinc_table(width, density) -> np.ndarray:
    """
    Generates a sinc table with the given width and density.
    The table will have values from -width to width, with density samples per unit.
    """
    beta = 14
    t = np.arange(-width * density, width * density) / density
    kernel = np.sinc(t) * np.kaiser(len(t), beta)
    return kernel


def lookup_table(kernel: np.ndarray, offset, density):
    """
    Look up a value from the kernel with linear interpolation.
    Offset is any value from -width to width, where width is the size of the kernel divided by density.
    """
    pos = len(kernel) // 2 + offset * density
    i = np.floor(pos).astype(int)
    frac = pos - i

    return kernel[i] * (1 - frac) + kernel[i + 1] * frac  # Linear interpolation so we don't lose info


@register("convert_sr")
def convert_sr(signal: Signal, new_sr: int) -> Signal:
    out = signal.clone()
    old_sr = signal.sr
    if new_sr == old_sr:
        return out  # No change needed

    num_samples = int(len(signal.data) * new_sr / old_sr)
    step = old_sr / new_sr
    out.sr = new_sr
    kernel = sinc_table(WIDTH, DENSITY)

    # The main loop:
    in_data = np.pad(signal.data, (WIDTH, WIDTH))
    out.data = np.zeros(num_samples)
    for i in range(num_samples):
        cursor = i * step  # Represents position in the old signal
        whole = int(cursor)  # The specific input signal we have passed
        frac = cursor - whole  # Represents fractional position between the input samples

        # We can't just simply repeat the old signal, we will lose information from the fractional position
        # To compensate, we will instead take a weighted sinc sum of nearby discrete points, with the weights being adjusted based on their distance from the cursor.
        window = in_data[whole - WIDTH : whole + WIDTH]
        weights = np.array([lookup_table(kernel, frac - np.arange(-WIDTH, WIDTH), DENSITY)])
        out.data[i] = np.dot(window, weights)

    return out


if __name__ == "__main__":
    x = sinc_table(WIDTH, DENSITY)
    y = lookup_table(x, 0, DENSITY)
    print(x[262144 // 2])
    print(y)
