"""
P300 BCI Game Configuration
===========================
Central configuration for all game parameters.

Based on research recommendations:
- Arrow size: ~1° visual angle (~100px at 60cm viewing distance)
- Flash duration: 100-125ms for optimal P300 response
- ISI: 100-150ms between flashes
- Eccentricity: ~5° from center for arrow placement

References:
- Ron-Angevin et al. (2019): Speller size optimization
- Takano et al. (2011): Green/Blue chromatic scheme
- Farwell & Donchin (1988): Original P300 paradigm
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from enum import Enum, auto
from pathlib import Path


# =============================================================================
# Enums
# =============================================================================

class Direction(Enum):
    """Movement directions for the game"""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    
    @classmethod
    def all(cls) -> List["Direction"]:
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]


class ColorScheme(Enum):
    """Available color schemes for arrows"""
    GRAY_WHITE = auto()    # Default: gray idle -> white flash
    GREEN_BLUE = auto()    # Comfort: blue idle -> green flash  
    INVERTED = auto()      # Light mode: light idle -> dark flash


class FlashPattern(Enum):
    """Stimulus presentation patterns"""
    RANDOM = auto()        # Random order each sequence
    SEQUENTIAL = auto()    # Fixed order: Up, Right, Down, Left


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class DisplayConfig:
    """Display and window settings"""
    width: int = 1920
    height: int = 1080
    fullscreen: bool = False
    fps: int = 60
    vsync: bool = True
    background_color: Tuple[int, int, int] = (10, 10, 10)


@dataclass  
class TimingConfig:
    """Stimulus timing parameters"""
    # Flash timing (in milliseconds)
    flash_duration_ms: int = 100      # How long arrow stays highlighted
    isi_ms: int = 125                 # Inter-stimulus interval
    
    # Sequence settings
    num_sequences: int = 10           # Repetitions per selection
    inter_sequence_pause_ms: int = 200
    
    # Flash pattern
    flash_pattern: FlashPattern = FlashPattern.RANDOM
    
    # Feedback
    feedback_duration_ms: int = 500
    
    @property
    def soa_ms(self) -> int:
        """Stimulus Onset Asynchrony"""
        return self.flash_duration_ms + self.isi_ms
    
    @property
    def flash_rate_hz(self) -> float:
        """Flash rate per arrow"""
        return 1000 / self.soa_ms


@dataclass
class ArrowConfig:
    """
    Arrow appearance settings.
    
    Arrow specifications (for 3072×1920 resolution):
    - Arrow icon box: 100×100 px
    - Triangle inside box: 80px length, 60px base width
    - Panel size: 200×200 px per arrow
    - Arrow-to-panel-edge margin: 50px (because 200 panel, 100 arrow)
    """
    size: int = 150                   # Arrow bounding box size in pixels
    
    # Triangle dimensions inside the bounding box
    triangle_length: int = 150        # Length in pointing direction
    triangle_base: int = 150          # Base width perpendicular to direction
    
    # Color scheme
    color_scheme: ColorScheme = ColorScheme.GRAY_WHITE
    
    # Gray/White scheme colors (default)
    idle_color: Tuple[int, int, int] = (128, 128, 128)
    flash_color: Tuple[int, int, int] = (255, 255, 255)
    
    # Panel behind arrows (200×200 px total)
    panel_size: int = 200             # Panel size in pixels
    panel_color: Tuple[int, int, int] = (0, 0, 0)
    panel_alpha: int = 153            # ~60% opacity (153/255)
    
    # Optional glow/border
    glow_thickness: int = 20          # Border thickness when flashing
    
    @property
    def panel_padding(self) -> int:
        """Padding from arrow edge to panel edge"""
        return (self.panel_size - self.size) // 2  # 50px with default values


@dataclass
class LayoutConfig:
    """
    Arrow layout configuration.
    
    Specifications (for 3072×1920):
    - Overlay window: 1150×1150 px (±50px), SQUARE, centered
    - Arrow offset from center: 475 px (to achieve 1150px with 200px panels)
    - Arrow size: 100×100 px
    """
    # Distance from screen center to arrow centers
    # For 1150px square: offset = (1150 - 200) / 2 = 475px
    horizontal_offset: int = 475      # Left/Right arrows
    vertical_offset: int = 475        # Up/Down arrows (same for square)
    
    # Keep-out spacing from panels to game elements
    keepout_margin: int = 50          # Recommended 50-90px
    
    def get_positions(self, screen_width: int, screen_height: int) -> dict:
        """Calculate arrow center positions"""
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
        """
        Get the overall bounding rectangle containing all arrow panels.
        
        Target: 1150×1150 px square, centered
        
        Returns:
            (left, top, width, height)
        """
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
    """Maze game settings"""
    # Colors (much duller grayscale to make arrows stand out)
    wall_color: Tuple[int, int, int] = (40, 40, 40)
    path_color: Tuple[int, int, int] = (25, 25, 25)
    player_color: Tuple[int, int, int] = (70, 70, 70)
    
    # Maze
    cell_size: int = 40
    
    # Player
    player_size: int = 30
    move_duration_ms: int = 300
    
    # Dullness level (1-5): higher = duller game elements, arrows stand out more
    # 5 = current brightness, 1 = very dull
    dullness: int = 5


@dataclass
class TriggerConfig:
    """EEG trigger/marker settings"""
    enabled: bool = True
    method: str = "file"              # "file", "lsl", "serial"
    
    # Trigger codes (must match MATLAB)
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
    """Main configuration combining all settings"""
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

# =============================================================================
# Real-time BCI configuration
# =============================================================================

BCI_MODE = True   # Set True to enable live EEG classification
                   # Set False to keep keyboard simulation (keys 1-4)

LSL_STREAM_TYPE = "EEG"
LSL_STREAM_NAME = None   # None = auto-discover; or set a specific name

# =============================================================================
# Model mode: "single" or "ensemble"
# =============================================================================

MODEL_MODE = "ensemble"    # "single" = one model, "ensemble" = weighted combination

# ── Single-model config (used when MODEL_MODE == "single") ───────────────────

MODEL_PATH = "models/single_trial_lda_best_model.joblib"

# ── Ensemble config (used when MODEL_MODE == "ensemble") ─────────────────────
#
# Each model entry maps a name to its saved .joblib artifact path.
# Every artifact must contain: "model", "feature_params", "cv_scores"
#
# Weights are from the best ensemble sweep (WeightedEnsemble12.ipynb).
# They must be in the same order as ENSEMBLE_MODEL_PATHS.

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

# Weights from the best ensemble (copy from your results)
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