"""
P300 BCI Game Configuration
===========================
Central configuration for all game parameters.

Based on research recommendations:
- Flash duration: 100-125ms for optimal P300 response
- ISI: 100-150ms between flashes

References:
- Ron-Angevin et al. (2019): Speller size optimization
- Takano et al. (2011): Green/Blue chromatic scheme
- Farwell & Donchin (1988): Original P300 paradigm
"""

from dataclasses import dataclass, field
from typing import Tuple, List
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
    - Arrow icon box: 150×150 px
    - Triangle inside box: 150px length, 150px base width
    - Panel size: 200×200 px per arrow
    - Arrow-to-panel-edge margin: 25px (because 200 panel, 150 arrow)
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


@dataclass
class LayoutConfig:
    """
    Arrow layout configuration.

    Specifications (for 3072×1920):
    - Arrow cluster: 1150×1150 px square, centered
    - Arrow offset from center: 475 px (to achieve 1150px with 200px panels)
    - Arrow size: 150×150 px
    """
    # Distance from screen center to arrow centers
    # For 1150px square: offset = (1150 - 200) / 2 = 475px
    horizontal_offset: int = 475      # Left/Right arrows
    vertical_offset: int = 475        # Up/Down arrows (same for square)

    def get_positions(self, screen_width: int, screen_height: int) -> dict:
        """Calculate arrow center positions"""
        cx, cy = screen_width // 2, screen_height // 2
        return {
            Direction.UP: (cx, cy - self.vertical_offset),
            Direction.DOWN: (cx, cy + self.vertical_offset),
            Direction.LEFT: (cx - self.horizontal_offset, cy),
            Direction.RIGHT: (cx + self.horizontal_offset, cy),
        }


@dataclass
class GameConfig:
    """Maze game settings"""
    # Dullness level (1-5): higher = duller game elements, arrows stand out more
    # 5 = current brightness, 1 = very dull
    dullness: int = 5


@dataclass
class TriggerConfig:
    """EEG trigger/marker settings"""
    enabled: bool = True
    method: str = "file"              # only "file" is implemented

    # Trigger codes
    TRIAL_START: int = 1
    TRIAL_END: int = 2
    FLASH_UP: int = 10
    FLASH_DOWN: int = 11
    FLASH_LEFT: int = 12
    FLASH_RIGHT: int = 13

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


# =============================================================================
# Real-time BCI configuration
# =============================================================================

# EEG stream format shared by the acquisition script, the mock stream, and the
# real-time classifier. Keep these in sync with the g.Nautilus configuration.
EEG_SAMPLING_RATE_HZ = 500
EEG_N_CHANNELS = 16

LSL_STREAM_TYPE = "EEG"
LSL_STREAM_NAME = None   # None = auto-discover; or set a specific name

# Model used for real-time classification
MODEL_PATH = "models/10trials_model.joblib"