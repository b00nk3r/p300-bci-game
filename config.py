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
    background_color: Tuple[int, int, int] = (20, 20, 20)


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
    """Arrow appearance settings"""
    size: int = 100                   # Arrow size in pixels
    
    # Color scheme
    color_scheme: ColorScheme = ColorScheme.GRAY_WHITE
    
    # Gray/White scheme colors
    idle_color: Tuple[int, int, int] = (128, 128, 128)
    flash_color: Tuple[int, int, int] = (255, 255, 255)
    
    # Panel behind arrows
    panel_padding: int = 50
    panel_color: Tuple[int, int, int] = (0, 0, 0)
    panel_alpha: int = 200


@dataclass
class LayoutConfig:
    """Arrow positioning"""
    # Offset from screen center (in pixels)
    horizontal_offset: int = 200
    vertical_offset: int = 150
    
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
    # Colors (grayscale to not interfere with arrows)
    wall_color: Tuple[int, int, int] = (100, 100, 100)
    path_color: Tuple[int, int, int] = (60, 60, 60)
    player_color: Tuple[int, int, int] = (150, 150, 150)
    
    # Maze
    cell_size: int = 40
    
    # Player
    player_size: int = 30
    move_duration_ms: int = 300


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