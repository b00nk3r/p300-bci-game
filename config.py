from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from enum import Enum, auto
from pathlib import Path


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    
    @classmethod
    def all(cls) -> List["Direction"]:
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]


class ColorScheme(Enum):
    GRAY_WHITE = auto()
    GREEN_BLUE = auto()
    INVERTED = auto()


class FlashPattern(Enum):
    RANDOM = auto()        # Random order each sequence
    SEQUENTIAL = auto()    # Fixed order: Up, Right, Down, Left


@dataclass
class DisplayConfig:
    width: int = 1920
    height: int = 1080
    fullscreen: bool = False
    fps: int = 60
    vsync: bool = True
    background_color: Tuple[int, int, int] = (10, 10, 10)


@dataclass  
class TimingConfig:
    flash_duration_ms: int = 100
    isi_ms: int = 125
    
    num_sequences: int = 10
    inter_sequence_pause_ms: int = 200
    
    flash_pattern: FlashPattern = FlashPattern.RANDOM
    
    feedback_duration_ms: int = 500
    
    @property
    def soa_ms(self) -> int:
        return self.flash_duration_ms + self.isi_ms
    
    @property
    def flash_rate_hz(self) -> float:
        return 1000 / self.soa_ms


@dataclass
class ArrowConfig:
    size: int = 150 # Arrow bounding box size in pixels
    
    # Triangle dimensions inside the bounding box
    triangle_length: int = 150
    triangle_base: int = 150
    
    color_scheme: ColorScheme = ColorScheme.GRAY_WHITE
    
    # Gray/White scheme colors (default)
    idle_color: Tuple[int, int, int] = (128, 128, 128)
    flash_color: Tuple[int, int, int] = (255, 255, 255)
    
    # Panel behind arrows
    panel_size: int = 200
    panel_color: Tuple[int, int, int] = (0, 0, 0)
    panel_alpha: int = 153
    
    glow_thickness: int = 20 # Border thickness when flashing
    
    @property
    def panel_padding(self) -> int:
        return (self.panel_size - self.size) // 2 # Padding from arrow edge to panel edge


@dataclass
class LayoutConfig:
    # Distance from screen center to arrow centers
    horizontal_offset: int = 475
    vertical_offset: int = 475
    
    # Keep-out spacing from panels to game elements
    keepout_margin: int = 50
    
    def get_positions(self, screen_width: int, screen_height: int) -> dict:
        cx, cy = screen_width // 2, screen_height // 2
        return {
            Direction.UP: (cx, cy - self.vertical_offset),
            Direction.DOWN: (cx, cy + self.vertical_offset),
            Direction.LEFT: (cx - self.horizontal_offset, cy),
            Direction.RIGHT: (cx + self.horizontal_offset, cy),
        }
    
    def get_overlay_rect(
        self, 
        screen_width: int, 
        screen_height: int, 
        panel_size: int = 200
    ) -> Tuple[int, int, int, int]:
        positions = self.get_positions(screen_width, screen_height)
        half_panel = panel_size // 2
        
        # Find bounds of all panels
        left = positions[Direction.LEFT][0] - half_panel
        right = positions[Direction.RIGHT][0] + half_panel
        top = positions[Direction.UP][1] - half_panel
        bottom = positions[Direction.DOWN][1] + half_panel
        
        return (left, top, right - left, bottom - top)


@dataclass
class GameConfig:
    wall_color: Tuple[int, int, int] = (40, 40, 40)
    path_color: Tuple[int, int, int] = (25, 25, 25)
    player_color: Tuple[int, int, int] = (70, 70, 70)
    
    cell_size: int = 40
    
    player_size: int = 30
    move_duration_ms: int = 300
    
    # Dullness level (1-5): higher = duller game elements, arrows stand out more
    dullness: int = 5


@dataclass
class TriggerConfig:
    enabled: bool = True
    method: str = "file" # "file", "lsl", "serial"
    
    # Trigger codes
    TRIAL_START: int = 1
    TRIAL_END: int = 2
    FLASH_UP: int = 10
    FLASH_DOWN: int = 11
    FLASH_LEFT: int = 12
    FLASH_RIGHT: int = 13
    SELECTION: int = 20
    
    # File output
    trigger_file: Path = Path("data/sessions/triggers.txt")


@dataclass
class Config:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    arrows: ArrowConfig = field(default_factory=ArrowConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    game: GameConfig = field(default_factory=GameConfig)
    triggers: TriggerConfig = field(default_factory=TriggerConfig)
    
    # Debug settings
    debug: bool = True
    show_fps: bool = True


# Default configuration instance
DEFAULT_CONFIG = Config()

BCI_MODE = True

LSL_STREAM_TYPE = "EEG"
LSL_STREAM_NAME = None # None = auto-discover


MODEL_MODE = "single"

MODEL_PATH = "models/10trials_model.joblib"

ENSEMBLE_MODEL_PATHS = {
    "1_LDA":          "models/single_trial_lda_best_model.joblib",
    "2_LR":           "models/single_trial_logistic_regression_best_model.joblib",
    "3_RF":           "models/single_trial_random_forest_best_model.joblib",
    "4_XGBoost":      "models/single_trial_xgboost_best_model.joblib",
    "5_LR_l1_bal":    "models/variant_lr_l1_bal_dec20.joblib",
    "6_LR_l2_C001":   "models/variant_lr_l2_C001_dec20.joblib",
    "7_LDA_dec40":    "models/variant_lda_dec40.joblib",
    "8_LDA_dec20":    "models/variant_lda_dec20.joblib",
    "9_LR_l2_bal":    "models/variant_lr_l2_C001_bal_dec30.joblib",
    "10_LDA_10Hz":    "models/variant_lda_10hz_dec30.joblib",
    "11_LDA_12ch":    "models/variant_lda_12ch_dec30.joblib",
    "12_SVM":         "models/single_trial_linear_svc_best_model.joblib",
}

ENSEMBLE_WEIGHTS = {
    "1_LDA":          0.131,
    "2_LR":           0.105,
    "3_RF":           0.274,
    "4_XGBoost":      0.011,
    "5_LR_l1_bal":    0.019,
    "6_LR_l2_C001":   0.053,
    "7_LDA_dec40":    0.069,
    "8_LDA_dec20":    0.040,
    "9_LR_l2_bal":    0.004,
    "10_LDA_10Hz":    0.097,
    "11_LDA_12ch":    0.170,
    "12_SVM":         0.027,
}