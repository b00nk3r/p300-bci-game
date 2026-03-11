DATA_DIR = "data/raw"

OUTPUT_DIR = "data/processed"


SR = 500
N_CHANNELS = 16


EPOCH_PRE_MS = 200
EPOCH_POST_MS = 800


BPF_LOW = 0.5
BPF_HIGH = 30.0
BPF_ORDER = 4


BASELINE_START_MS = -100
BASELINE_END_MS = 0


ARTIFACT_PP_THRESHOLD = 150.0


BAD_CH_LOW_FACTOR = 0.01
BAD_CH_HIGH_FACTOR = 4.0


DIRECTION_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
}


DIRECTION_NAMES = {v: k for k, v in DIRECTION_MAP.items()}

PARAM_WINDOWS = [
    (0, 800),
    (0, 600),
    (200, 600),
    (200, 800),
    (100, 700),
    (0, 650),
    (0,700),
]

PARAM_DEC_WINDOWS = [20, 24, 30, 40]