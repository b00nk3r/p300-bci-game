"""
Real-time BCI configuration.
All preprocessing parameters MUST match the offline training pipeline.
"""

# --- EEG Hardware ---
SR = 500                    # Sampling rate in Hz
N_CHANNELS = 16             # Number of EEG channels

# --- Preprocessing (must match training) ---
BPF_LOW = 0.5               # Bandpass filter low cutoff (Hz)
BPF_HIGH = 30.0             # Bandpass filter high cutoff (Hz)
BPF_ORDER = 4               # Butterworth filter order

BASELINE_START_MS = -100     # Baseline correction window start (ms relative to stimulus)
BASELINE_END_MS = 0          # Baseline correction window end

ARTIFACT_PP_THRESHOLD = 150.0  # Peak-to-peak rejection threshold (µV)

BAD_CH_LOW_FACTOR = 0.01    # Bad channel detection: low std threshold
BAD_CH_HIGH_FACTOR = 4.0    # Bad channel detection: high std threshold

# --- Epoching ---
EPOCH_PRE_MS = 200           # Pre-stimulus epoch window (ms)
EPOCH_POST_MS = 800          # Post-stimulus epoch window (ms)

# --- Feature Extraction (loaded from model artifact at runtime) ---
# These are defaults; the actual values come from the saved model file
DEFAULT_WINDOW_START_MS = 0
DEFAULT_WINDOW_END_MS = 800
DEFAULT_DEC_WINDOW = 20
DEFAULT_DEC_STEP = 10

# --- Real-time Buffer ---
BUFFER_DURATION_S = 60       # Circular buffer duration in seconds

# --- Model ---
MODEL_PATH = "models/single_trial_lda_best_model.joblib"

# --- Direction Mapping ---
DIRECTION_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
}
DIRECTION_NAMES = {v: k for k, v in DIRECTION_MAP.items()}
