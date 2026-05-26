# Register all DSP functions here
from .modify import add_sines
from .modify import add_square
from .modify import fir_filter
from .modify import gain
from .modify import generate_sines
from .modify import generate_square
from .modify import pitch_shift
from .modify import rolling_avg
from .modify import generate_noise
from .modify import convert_sr
from .modify import truncate
from .modify import low_pass
from .modify import high_pass

from .analyze import duration
from .analyze import max_frequency
from .analyze import max_volume
from .analyze import sample_count
from .analyze import sample_rate
from .analyze import top_frequencies
