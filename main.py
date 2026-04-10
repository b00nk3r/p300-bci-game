#!/usr/bin/env python3
"""
P300 BCI Game - Main Entry Point
================================
Salem State University Capstone Project

A Brain-Computer Interface game using P300 evoked potentials
to control a maze character through flashing arrow stimuli.

Controls:
    SPACE  - Run one full calibration pass (data collection mode)
    S      - Open settings panel (TODO)
    D      - Toggle debug overlay
    ESC    - Quit
    Arrows - Live BCI target labels / manual movement fallback
    1-4    - Simulate BCI selection (Up/Down/Left/Right)

Usage:
    python main.py
    python main.py --fullscreen
    python main.py --debug
    python main.py --width 1280 --height 720
"""

import sys
import argparse
import time
import random
from enum import Enum, auto

import pygame

from config import Config, Direction
from src.stimulus.arrow_manager import ArrowManager, SelectionState, SelectionResult
from src.ui.settings_panel import SettingsPanel, SettingsValues
from src.game.game_manager import GameManager, GameManagerConfig, GameState
from src.data.session_logger import SessionLogger


# Design resolution - the resolution all game elements are laid out for.
# The game always renders internally at this resolution, then scales to fit
# whatever window size the user requests.
DESIGN_WIDTH = 3072
DESIGN_HEIGHT = 1920


class CalibrationStage(Enum):
    """Stages for one full calibration run."""
    IDLE = auto()
    INSTRUCTION = auto()
    FLASHING = auto()
    BREAK = auto()
    WAITING = auto()       # Paused after classification, waiting for next target input


class Application:
    """
    Main application class for P300 BCI Game.
    
    Manages:
    - Pygame initialization and main loop
    - Arrow stimulus system (via ArrowManager)
    - Game state (maze - TODO)
    - User input handling
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.clock = None
        self.screen = None
        
        # Components
        self.arrow_manager: ArrowManager = None
        self.settings_panel: SettingsPanel = None
        self.game_manager: GameManager = None
        self.session_logger: SessionLogger = None
        self.classifier = None
        
        # Resolution-independent rendering
        self.render_surface = None  # Internal surface at design resolution
        self.scale_factor = 1.0
        self.render_offset = (0, 0)
        self.scaled_size = (DESIGN_WIDTH, DESIGN_HEIGHT)
        
        # Display info
        self.num_displays = 1
        self.display_index = 0
        self.display_override = False  # True if manually set via command line
        
        # State
        self.show_debug = config.debug
        self.last_selection: SelectionResult = None
        
        # Fonts (created on initialize)
        self.font_large = None
        self.font_medium = None
        self.font_small = None

        # Data-recording / calibration run state
        self.calibration_stage = CalibrationStage.IDLE
        self.calibration_phase_order = []
        self.calibration_phase_index = 0
        self.calibration_stage_start_time = 0.0
        self.calibration_run_start_time = 0.0

        self.calibration_flash_states = {d: False for d in Direction.all()}
        self.calibration_flash_plan = []
        self.calibration_flash_index = 0
        self.calibration_current_flash = None
        self.calibration_current_flash_end_time = 0.0
        self.calibration_next_flash_time = 0.0

        self.calibration_instruction_ms = 2000
        self.calibration_break_ms = 2000

        # Warm-up state (throwaway trials to stabilize feature normalization)
        self.warmup_remaining = 0
        self.warmup_total = 0
        
    def initialize(self):
        """Initialize pygame and all components"""
        pygame.init()
        
        # Detect available displays
        self.num_displays = pygame.display.get_num_displays()
        
        # Auto-select display if not manually overridden
        if not self.display_override:
            # Use second display if available, otherwise use primary
            # This allows the app to work on both single and dual-display setups
            self.display_index = 1 if self.num_displays > 1 else 0
        
        # Create display
        flags = pygame.DOUBLEBUF
        if self.config.display.fullscreen:
            flags |= pygame.FULLSCREEN
            
        self.screen = pygame.display.set_mode(
            (self.config.display.width, self.config.display.height),
            flags,
            display=self.display_index
        )
        pygame.display.set_caption("P300 BCI Game - Salem State University")
        
        self.clock = pygame.time.Clock()
        
        # Create render surface at design resolution
        self.render_surface = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))
        self._calculate_scaling()
        
        # Create fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize arrow manager
        self._init_arrow_manager()
        
        # Initialize BCI classifier (if enabled)
        self._init_classifier()
        
        # Initialize session logger
        self._init_session_logger()
        
        # Initialize settings panel
        self._init_settings_panel()
        
        # Initialize game manager
        self._init_game_manager()
        
        # Print startup info
        self._print_startup_info()
        
    def _calculate_scaling(self):
        """Calculate scale factor and offset for resolution-independent rendering.
        
        Maps the design resolution (DESIGN_WIDTH x DESIGN_HEIGHT) to the actual
        window, maintaining aspect ratio with letterboxing if needed.
        """
        window_w = self.config.display.width
        window_h = self.config.display.height
        
        scale_x = window_w / DESIGN_WIDTH
        scale_y = window_h / DESIGN_HEIGHT
        self.scale_factor = min(scale_x, scale_y)
        
        # Calculate scaled dimensions and centering offset (letterboxing)
        scaled_w = int(DESIGN_WIDTH * self.scale_factor)
        scaled_h = int(DESIGN_HEIGHT * self.scale_factor)
        self.scaled_size = (scaled_w, scaled_h)
        self.render_offset = (
            (window_w - scaled_w) // 2,
            (window_h - scaled_h) // 2
        )
        
    def _transform_event(self, event):
        """Transform mouse coordinates from window space to design resolution space."""
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            x, y = event.pos
            # Remove letterbox offset and scale to design resolution
            x = int((x - self.render_offset[0]) / self.scale_factor)
            y = int((y - self.render_offset[1]) / self.scale_factor)
            
            attrs = dict(event.__dict__)
            attrs['pos'] = (x, y)
            if event.type == pygame.MOUSEMOTION and hasattr(event, 'rel'):
                rx, ry = event.rel
                attrs['rel'] = (int(rx / self.scale_factor), int(ry / self.scale_factor))
            return pygame.event.Event(event.type, **attrs)
        return event
        
    def _init_arrow_manager(self):
        """Initialize the arrow stimulus system"""
        self.arrow_manager = ArrowManager(self.config)
        self.arrow_manager.initialize(
            DESIGN_WIDTH, 
            DESIGN_HEIGHT
        )
        
        # Set callbacks
        self.arrow_manager.set_callbacks(
            on_selection_complete=self._on_selection_complete,
            on_state_change=self._on_state_change,
            on_flash_start=self._on_flash_start,
            on_flash_end=self._on_flash_end,
        )

    def _init_classifier(self):
        """Initialize the real-time BCI classifier if BCI_MODE is enabled."""
        from config import BCI_MODE, LSL_STREAM_TYPE, LSL_STREAM_NAME

        self.classifier = None
        if BCI_MODE:
            from realtime_classifier import create_classifier
            self.classifier = create_classifier(
                stream_type=LSL_STREAM_TYPE,
                stream_name=LSL_STREAM_NAME,
            )
            if not self.classifier.start():
                print("WARNING: Could not connect to EEG stream. "
                      "Falling back to keyboard simulation.")
                self.classifier = None

        self.arrow_manager.classifier = self.classifier

    def _init_session_logger(self):
        """Initialize the session logger for data recording"""
        self.session_logger = SessionLogger(output_dir="data/sessions")
        
    def _init_settings_panel(self):
        """Initialize the settings panel"""
        self.settings_panel = SettingsPanel(
            DESIGN_WIDTH,
            DESIGN_HEIGHT
        )
        
        # Set current values from config
        values = SettingsValues.from_config(self.config)
        self.settings_panel.set_values(values)
        
        # Set callbacks
        self.settings_panel.set_callbacks(
            on_apply=self._on_settings_apply,
            on_cancel=self._on_settings_cancel,
        )
        
    def _init_game_manager(self):
        """Initialize the maze game"""
        # Get arrow panel size to calculate appropriate maze size
        panel_rect = self.arrow_manager.get_panel_rect()
        
        # =================================================================
        # MAZE SIZE CONFIGURATION
        # =================================================================
        # Cell size determines how large each maze cell appears on screen.
        # Larger cells = fewer cells = simpler maze (easier to play)
        # Smaller cells = more cells = complex maze (harder to play)
        #
        # Fixed grid target:
        # - 10 columns x 6 rows
        # - Larger cells for easier play
        #
        # To adjust: change target_*_cells below
        # =================================================================
        
        target_width_cells = 10
        target_height_cells = 6
        cell_size = min(
            DESIGN_WIDTH // target_width_cells,
            DESIGN_HEIGHT // target_height_cells
        )
        
        # Calculate maze dimensions to fill screen
        margin = 0  # No margin - fill entire screen
        
        maze_width_cells = target_width_cells
        maze_height_cells = target_height_cells
        
        # For corridor mode, we don't need odd dimensions
        # Use corridor mode for large cells (simpler paths)
        use_corridors = cell_size >= 128
        
        game_config = GameManagerConfig(
            base_maze_width=maze_width_cells,
            base_maze_height=maze_height_cells,
            max_maze_width=maze_width_cells,  # Don't grow beyond screen
            max_maze_height=maze_height_cells,
            maze_growth_per_level=0,  # Keep same size, just regenerate
            base_collectibles=5,  # Fixed donut count per level
            collectibles_per_level=0,
            max_collectibles=5,
            cell_size=cell_size,
            use_corridors=use_corridors,  # Use corridor mode for large cells
        )
        
        self.game_manager = GameManager(game_config)
        
        # Get arrow positions for plus-shaped forbidden zone
        arrow_positions = self.config.layout.get_positions(
            DESIGN_WIDTH,
            DESIGN_HEIGHT
        )
        
        self.game_manager.initialize(
            DESIGN_WIDTH,
            DESIGN_HEIGHT,
            panel_rect,
            arrow_positions
        )
        
        # Set callbacks
        self.game_manager.set_callbacks(
            on_level_complete=self._on_level_complete,
            on_item_collected=self._on_item_collected,
        )
        
        # Set initial dullness from config
        self.game_manager.set_dullness(self.config.game.dullness)
        
        # Start the game
        self.game_manager.start_game()
        
    def _print_startup_info(self):
        """Print configuration info to console"""
        print()
        print("=" * 50)
        print("P300 BCI Game - Salem State University")
        print("=" * 50)
        print()
        print("Display:")
        print(f"  Available displays: {self.num_displays}")
        display_type = "(primary)" if self.display_index == 0 else "(secondary)"
        override_text = " [manual]" if self.display_override else " [auto]"
        print(f"  Using display: {self.display_index + 1} {display_type}{override_text}")
        print()
        print("Configuration:")
        print(f"  Window: {self.config.display.width}x{self.config.display.height}")
        print(f"  Design: {DESIGN_WIDTH}x{DESIGN_HEIGHT}")
        print(f"  Scale: {self.scale_factor:.3f}x")
        print(f"  Fullscreen: {self.config.display.fullscreen}")
        print()
        print("Timing:")
        print(f"  Flash duration: {self.config.timing.flash_duration_ms}ms")
        print(f"  ISI: {self.config.timing.isi_ms}ms")
        print(f"  SOA: {self.config.timing.soa_ms}ms")
        print(f"  Flash rate: {self.config.timing.flash_rate_hz:.1f}Hz per arrow")
        print(f"  Sequences: {self.config.timing.num_sequences}")
        print()
        from config import BCI_MODE
        print(f"BCI Mode: {'ENABLED' if BCI_MODE else 'DISABLED (keyboard simulation)'}")
        if self.classifier:
            print(f"  Classifier: connected")
        elif BCI_MODE:
            print(f"  Classifier: FAILED to connect (keyboard fallback)")
        print()
        print("Controls:")
        if self._is_live_bci_mode():
            print("  Arrows - Set target and start a labeled BCI trial")
            print("           (movement still comes from the model)")
            print("  SPACE  - Not used for live BCI trials")
        else:
            print("  SPACE  - Run one full calibration pass")
            print("  Arrows - Manual movement (testing)")
        print("  S      - Open settings panel")
        print("  D      - Toggle debug info")
        print("  R      - Restart current level")
        print("  N      - Skip to next level")
        print("  1-4    - Simulate BCI selection (Up/Down/Left/Right)")
        print("  ESC    - Quit")
        print()
        print("=" * 50)
        print()
        
    def run(self):
        """Main game loop"""
        self.running = True
        
        while self.running:
            # Handle events
            self._handle_events()
            
            # Update
            self._update()
            
            # Draw
            self._draw()
            
            # Cap framerate
            self.clock.tick(self.config.display.fps)
            
        self._cleanup()
        
    def _handle_events(self):
        """Process input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            
            # Transform mouse coordinates from window space to design space
            event = self._transform_event(event)
                
            # Let settings panel handle events first (if visible)
            if self.settings_panel and self.settings_panel.is_visible:
                if self.settings_panel.handle_event(event):
                    continue  # Event consumed by panel
                    
            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)

    def _is_live_bci_mode(self) -> bool:
        """Whether live EEG classification is currently available."""
        from config import BCI_MODE

        return BCI_MODE and self.classifier is not None

    @staticmethod
    def _direction_from_key(key: int):
        """Map arrow keys to directions."""
        key_map = {
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
        }
        return key_map.get(key)
                
    def _handle_keydown(self, key: int):
        """Handle keyboard input"""
        # Quit
        if key == pygame.K_ESCAPE:
            self.running = False
            
        # Run warm-up / calibration pass / resume from WAITING
        elif key == pygame.K_SPACE:
            if self._is_live_bci_mode():
                if self.calibration_stage in (CalibrationStage.IDLE, CalibrationStage.WAITING):
                    self._start_warmup()
                else:
                    print("BCI trial already running")
            elif self.calibration_stage == CalibrationStage.WAITING:
                self._advance_calibration_phase()
            else:
                self._start_calibration_run()
            
        # Toggle debug
        elif key == pygame.K_d:
            self.show_debug = not self.show_debug
            
        # Toggle settings panel
        elif key == pygame.K_s:
            self._toggle_settings()
            
        # Game controls
        elif key == pygame.K_r:
            # Restart level
            if self.game_manager:
                self.game_manager.restart_level()
                print("Level restarted")
        elif key == pygame.K_n:
            # Next level (for testing)
            if self.game_manager:
                self.game_manager.next_level()
                print(f"Advanced to level {self.game_manager.stats.level}")
            
        # Simulate selections (for testing)
        elif key == pygame.K_1:
            self._simulate_selection(Direction.UP)
        elif key == pygame.K_2:
            self._simulate_selection(Direction.DOWN)
        elif key == pygame.K_3:
            self._simulate_selection(Direction.LEFT)
        elif key == pygame.K_4:
            self._simulate_selection(Direction.RIGHT)

        else:
            direction = self._direction_from_key(key)
            if direction is not None:
                if self._handle_live_bci_target_key(direction):
                    return
                self._manual_move(direction)

    def _handle_live_bci_target_key(self, direction: Direction) -> bool:
        """Use arrow keys as target labels when live EEG mode is active."""
        if not self._is_live_bci_mode():
            return False

        if self.arrow_manager.is_active:
            print("Legacy BCI selection already running")
            return True

        if self.calibration_stage in (CalibrationStage.IDLE, CalibrationStage.WAITING):
            self._start_live_bci_trial(direction)
            return True

        if self._is_calibration_active():
            print("BCI trial already running")
            return True

        return False
            
    def _is_calibration_active(self) -> bool:
        """Whether a calibration run is currently active."""
        return self.calibration_stage != CalibrationStage.IDLE

    def _is_warmup_active(self) -> bool:
        """Whether a warm-up sequence is in progress."""
        return self.warmup_remaining > 0

    def _start_warmup(self):
        """Begin the warm-up sequence to stabilize feature normalization."""
        from config import WARMUP_TRIALS
        if not self._is_live_bci_mode() or WARMUP_TRIALS <= 0:
            print("Press a target arrow to start the next BCI trial.")
            return
        if self.warmup_total > 0:
            print("Press a target arrow to start the next BCI trial.")
            return
        self.warmup_remaining = WARMUP_TRIALS
        self.warmup_total = WARMUP_TRIALS
        print(f"\n{'='*50}")
        print(f"WARM-UP: {WARMUP_TRIALS} trials to stabilize")
        print(f"feature normalization. Attend each arrow.")
        print(f"{'='*50}\n")
        self._start_next_warmup_trial()

    def _start_next_warmup_trial(self):
        """Pick a random direction and begin one warm-up trial."""
        target = random.choice(Direction.all())

        self.calibration_phase_order = [target]
        self.calibration_phase_index = 0
        self.calibration_run_start_time = time.perf_counter()

        if self.classifier:
            self.classifier.clear_events()

        self.arrow_manager.triggers.stop_session()
        self.arrow_manager.triggers.start_session(
            flash_duration_ms=self.config.timing.flash_duration_ms,
            isi_ms=self.config.timing.isi_ms,
            soa_ms=self.config.timing.soa_ms,
            num_sequences=self.config.timing.num_sequences,
            inter_sequence_pause_ms=0,
            flash_pattern="RANDOM",
            color_scheme=self.config.arrows.color_scheme.name,
            flash_rate_hz=self.config.timing.flash_rate_hz,
        )
        self.arrow_manager.triggers.set_current_target(target)
        self.arrow_manager.triggers.send_trial_start()

        self._set_calibration_idle_arrows()
        self._start_instruction_stage()

    def _finish_warmup(self):
        """End the warm-up sequence and return to idle."""
        print(f"\n{'='*50}")
        print("Warm-up complete! Feature statistics stabilized.")
        print(f"{'='*50}\n")
        print("Press a target arrow to start playing.")

        self.warmup_remaining = 0
        self.warmup_total = 0
        self.calibration_stage = CalibrationStage.IDLE
        self.calibration_phase_order = []
        self.calibration_phase_index = 0
        self._set_calibration_idle_arrows()

    def _start_calibration_run(self):
        """Start one full data-recording calibration run."""
        if self._is_calibration_active():
            print("Calibration already running")
            return

        # Keep behavior predictable if legacy selection is active.
        if self.arrow_manager.is_active:
            self.arrow_manager.stop_selection()

        self.calibration_phase_order = Direction.all()
        random.shuffle(self.calibration_phase_order)
        self.calibration_phase_index = 0
        self.calibration_run_start_time = time.perf_counter()

        # Start one session for the whole 4-phase run.
        if self.session_logger:
            self.session_logger.start_session(
                flash_duration_ms=self.config.timing.flash_duration_ms,
                isi_ms=self.config.timing.isi_ms,
                num_sequences=self.config.timing.num_sequences,
                inter_sequence_pause_ms=0,
                flash_pattern="RANDOM",
                color_scheme=self.config.arrows.color_scheme.name,
            )

        self.arrow_manager.triggers.start_session(
            flash_duration_ms=self.config.timing.flash_duration_ms,
            isi_ms=self.config.timing.isi_ms,
            soa_ms=self.config.timing.soa_ms,
            num_sequences=self.config.timing.num_sequences,
            inter_sequence_pause_ms=0,
            flash_pattern="RANDOM",
            color_scheme=self.config.arrows.color_scheme.name,
            flash_rate_hz=self.config.timing.flash_rate_hz,
        )
        first_target = self.calibration_phase_order[self.calibration_phase_index]
        self.arrow_manager.triggers.set_current_target(first_target)
        self.arrow_manager.triggers.send_trial_start()

        self._set_calibration_idle_arrows()
        self._start_instruction_stage()
        print("Calibration run started")

    def _start_live_bci_trial(self, target: Direction):
        """Start one labeled BCI gameplay trial for the given target arrow."""
        if not self._is_live_bci_mode():
            return

        if self._is_calibration_active() and self.calibration_stage not in (
            CalibrationStage.IDLE,
            CalibrationStage.WAITING,
        ):
            print("BCI trial already running")
            return

        if self.arrow_manager.is_active:
            self.arrow_manager.stop_selection()

        self.calibration_phase_order = [target]
        self.calibration_phase_index = 0
        self.calibration_run_start_time = time.perf_counter()

        if self.classifier:
            self.classifier.clear_events()

        if self.session_logger and self.session_logger.is_active:
            self.session_logger.cancel_session()

        self.arrow_manager.triggers.stop_session()

        if self.session_logger:
            self.session_logger.start_session(
                flash_duration_ms=self.config.timing.flash_duration_ms,
                isi_ms=self.config.timing.isi_ms,
                num_sequences=self.config.timing.num_sequences,
                inter_sequence_pause_ms=0,
                flash_pattern="RANDOM",
                color_scheme=self.config.arrows.color_scheme.name,
                target_direction=target,
            )

        self.arrow_manager.triggers.start_session(
            flash_duration_ms=self.config.timing.flash_duration_ms,
            isi_ms=self.config.timing.isi_ms,
            soa_ms=self.config.timing.soa_ms,
            num_sequences=self.config.timing.num_sequences,
            inter_sequence_pause_ms=0,
            flash_pattern="RANDOM",
            color_scheme=self.config.arrows.color_scheme.name,
            flash_rate_hz=self.config.timing.flash_rate_hz,
        )
        self.arrow_manager.triggers.set_current_target(target)
        self.arrow_manager.triggers.send_trial_start()

        self._set_calibration_idle_arrows()
        self._start_flashing_stage()
        print(f"BCI trial started - target {target.value.upper()}")

    def _set_calibration_idle_arrows(self):
        """Set all calibration arrows to non-flashing state."""
        self.calibration_flash_states = {d: False for d in Direction.all()}
        self.calibration_current_flash = None
        self.calibration_current_flash_end_time = 0.0

    def _start_instruction_stage(self):
        """Start instruction display for current attended arrow."""
        attended = self.calibration_phase_order[self.calibration_phase_index]
        self.arrow_manager.triggers.set_current_target(attended)
        total_phases = len(self.calibration_phase_order)

        from config import BCI_MODE
        if BCI_MODE and self.classifier and not self._is_warmup_active():
            print(
                f"Phase {self.calibration_phase_index + 1}/{total_phases} - "
                f"Flashing {attended.value.upper()}"
            )
            self._start_flashing_stage()
        else:
            self.calibration_stage = CalibrationStage.INSTRUCTION
            self.calibration_stage_start_time = time.perf_counter()
            self._set_calibration_idle_arrows()
            if self._is_warmup_active():
                warmup_num = self.warmup_total - self.warmup_remaining + 1
                print(
                    f"Warm-up {warmup_num}/{self.warmup_total} - "
                    f"ATTEND {attended.value.upper()}"
                )
            else:
                print(
                    f"Phase {self.calibration_phase_index + 1}/{total_phases} - "
                    f"ATTEND {attended.value.upper()}"
                )

    def _start_flashing_stage(self):
        """Start flashing stage for current attended arrow phase."""
        if self.calibration_phase_order and self.calibration_phase_index < len(self.calibration_phase_order):
            attended = self.calibration_phase_order[self.calibration_phase_index]
            self.arrow_manager.triggers.set_current_target(attended)

        self.calibration_stage = CalibrationStage.FLASHING
        self.calibration_stage_start_time = time.perf_counter()
        self._set_calibration_idle_arrows()

        self.calibration_flash_plan = []
        for _ in range(self.config.timing.num_sequences):
            sequence_order = Direction.all()
            random.shuffle(sequence_order)
            self.calibration_flash_plan.extend(sequence_order)

        self.calibration_flash_index = 0
        self.calibration_next_flash_time = time.perf_counter()  # Start immediately

    def _start_break_stage(self):
        """Start blank/neutral break between attended-arrow phases."""
        self.calibration_stage = CalibrationStage.BREAK
        self.calibration_stage_start_time = time.perf_counter()
        self.arrow_manager.triggers.set_current_target(None)
        self._set_calibration_idle_arrows()

    def _start_waiting_stage(self):
        """Pause after classification until the next target is chosen."""
        self.calibration_stage = CalibrationStage.WAITING
        self.calibration_stage_start_time = time.perf_counter()
        self.arrow_manager.triggers.set_current_target(None)
        self._set_calibration_idle_arrows()
        if self._is_live_bci_mode():
            print("Waiting for target arrow (UP/DOWN/LEFT/RIGHT)...")
        else:
            print("Waiting for SPACE to start next flash round...")

    def _finalize_live_bci_trial(self, selected_direction: Direction = None):
        """Close per-trial logs after a live BCI gameplay attempt."""
        self.arrow_manager.triggers.send_trial_end()
        self.arrow_manager.triggers.stop_session()

        if self.session_logger and self.session_logger.is_active:
            self.session_logger.end_session(selected_direction=selected_direction)

    def _advance_calibration_phase(self):
        """Advance to the next attended direction or finish the run."""
        self.calibration_phase_index += 1

        if self.calibration_phase_index >= len(self.calibration_phase_order):
            from config import BCI_MODE
            if BCI_MODE and self.classifier:
                # Loop continuously in BCI mode, but wait for SPACE between rounds.
                self.calibration_phase_index = 0
                random.shuffle(self.calibration_phase_order)
                self._start_instruction_stage()
            else:
                self._finish_calibration_run()
            return

        self._start_instruction_stage()

    def _finish_calibration_run(self, cancelled: bool = False):
        """Finalize calibration run, close logs, and return to idle."""
        was_active = self._is_calibration_active()

        self.calibration_stage = CalibrationStage.IDLE
        self.calibration_phase_order = []
        self.calibration_phase_index = 0
        self.calibration_stage_start_time = 0.0
        self.calibration_flash_plan = []
        self.calibration_flash_index = 0
        self.calibration_next_flash_time = 0.0
        self.warmup_remaining = 0
        self.warmup_total = 0
        self._set_calibration_idle_arrows()

        if was_active:
            self.arrow_manager.triggers.set_current_target(None)
            self.arrow_manager.triggers.send_trial_end()
            self.arrow_manager.triggers.stop_session()

        if self.session_logger and self.session_logger.is_active:
            if cancelled:
                self.session_logger.cancel_session()
            else:
                self.session_logger.end_session()

        if cancelled:
            print("Calibration run cancelled")
        else:
            print("Calibration run complete - waiting for SPACE")

    def _update_calibration_run(self):
        """Advance the calibration run state machine."""
        now = time.perf_counter()
        elapsed_stage_ms = (now - self.calibration_stage_start_time) * 1000.0

        if self.calibration_stage == CalibrationStage.INSTRUCTION:
            if elapsed_stage_ms >= self.calibration_instruction_ms:
                self._start_flashing_stage()
            return

        if self.calibration_stage == CalibrationStage.FLASHING:
            self._update_calibration_flashing(now)
            return

        if self.calibration_stage == CalibrationStage.WAITING:
            return

        if self.calibration_stage == CalibrationStage.BREAK:
            if elapsed_stage_ms >= self.calibration_break_ms:
                if self._is_warmup_active():
                    self._start_next_warmup_trial()
                else:
                    self._advance_calibration_phase()

    def _update_calibration_flashing(self, now: float):
        """Update flash timing for the current attended-arrow phase."""
        # End active flash if duration has elapsed.
        if (
            self.calibration_current_flash is not None
            and now >= self.calibration_current_flash_end_time
        ):
            direction = self.calibration_current_flash
            sequence = self.calibration_flash_index // 4

            self.calibration_flash_states[direction] = False
            timestamp_ms = (now - self.calibration_run_start_time) * 1000.0
            self._on_flash_end(direction, sequence, timestamp_ms)

            self.calibration_current_flash = None
            self.calibration_flash_index += 1

            if self.calibration_flash_index >= len(self.calibration_flash_plan):
                selected_direction = None
                if self.classifier:
                    result = self.classifier.classify_trial()
                    if result:
                        selected_direction = Direction(result["direction"])
                        if self._is_warmup_active():
                            warmup_num = self.warmup_total - self.warmup_remaining + 1
                            print(
                                f"[Warm-up {warmup_num}/{self.warmup_total}] "
                                f"Result: {selected_direction.value.upper()} "
                                f"(confidence={result['confidence']:.3f}, "
                                f"epochs={result['n_epochs_used']}/{result['n_epochs_total']})"
                            )
                        else:
                            print(
                                f"BCI Selection: {selected_direction.value.upper()} "
                                f"(confidence={result['confidence']:.3f}, "
                                f"epochs={result['n_epochs_used']}/{result['n_epochs_total']})"
                            )
                            if self.game_manager and self.game_manager.can_accept_input:
                                moved = self.game_manager.move_player(selected_direction)
                                if not moved:
                                    print("  (blocked by wall)")
                    else:
                        print("BCI classification failed for this phase.")
                    self.classifier.clear_events()

                if self._is_warmup_active():
                    self.arrow_manager.triggers.send_trial_end()
                    self.arrow_manager.triggers.stop_session()
                    self.warmup_remaining -= 1
                    if self.warmup_remaining > 0:
                        self._start_break_stage()
                    else:
                        self._finish_warmup()
                elif self._is_live_bci_mode():
                    self._finalize_live_bci_trial(selected_direction)
                    self._start_waiting_stage()
                else:
                    self._start_break_stage()
                return

            self.calibration_next_flash_time = now + (self.config.timing.isi_ms / 1000.0)

        # Start next flash when ISI has elapsed and no flash is active.
        if (
            self.calibration_current_flash is None
            and self.calibration_flash_index < len(self.calibration_flash_plan)
            and now >= self.calibration_next_flash_time
        ):
            direction = self.calibration_flash_plan[self.calibration_flash_index]
            sequence = self.calibration_flash_index // 4

            self._set_calibration_idle_arrows()
            self.calibration_flash_states[direction] = True
            self.calibration_current_flash = direction
            self.calibration_current_flash_end_time = (
                now + (self.config.timing.flash_duration_ms / 1000.0)
            )

            timestamp_ms = (now - self.calibration_run_start_time) * 1000.0
            self.arrow_manager.triggers.send_flash(direction)

            if self.arrow_manager.classifier is not None:
                from pylsl import local_clock
                self.arrow_manager.classifier.record_flash(
                    direction=direction.value,
                    timestamp=local_clock(),
                )

            self._on_flash_start(direction, sequence, timestamp_ms)
            
    def _simulate_selection(self, direction: Direction):
        """Simulate a classifier result (for testing)"""
        if self.arrow_manager.state in (SelectionState.FLASHING, SelectionState.PROCESSING):
            self.arrow_manager.simulate_selection(direction)
            print(f"Simulated selection: {direction.value}")
        else:
            print("Simulation only works during legacy selection flashing")
            
    def _manual_move(self, direction: Direction):
        """Handle manual movement (for testing game without BCI)"""
        if self.game_manager and self.game_manager.can_accept_input:
            moved = self.game_manager.move_player(direction)
            if moved:
                print(f"Manual move: {direction.value}")
            else:
                print(f"Manual move: {direction.value} (blocked)")
        
    def _toggle_settings(self):
        """Toggle the settings panel"""
        if self.settings_panel.is_visible:
            self.settings_panel.hide()
        else:
            # Don't open settings during active selection
            if self.arrow_manager.is_active or self._is_calibration_active():
                print("Cannot open settings during active selection")
                return
                
            # Update panel with current values
            values = SettingsValues.from_config(self.config)
            self.settings_panel.set_values(values)
            self.settings_panel.show()
            print("Settings panel opened")
            
    def _on_settings_apply(self, values: SettingsValues):
        """Called when settings are applied"""
        print(f"Settings applied:")
        print(f"  Flash: {values.flash_duration_ms}ms")
        print(f"  ISI: {values.isi_ms}ms")
        print(f"  Sequences: {values.num_sequences}")
        print(f"  Dullness: {values.dullness}")
        print(f"  Color: {values.color_scheme.name}")
        print(f"  SOA: {values.soa_ms}ms ({values.flash_rate_hz:.1f}Hz)")
        
        # Apply to config
        values.apply_to_config(self.config)
        
        # Reinitialize arrow manager with new settings
        self.arrow_manager.shutdown()
        self._init_arrow_manager()
        self.arrow_manager.classifier = self.classifier
        
        # Update game manager dullness
        if self.game_manager:
            self.game_manager.set_dullness(self.config.game.dullness)
        
    def _on_settings_cancel(self):
        """Called when settings are cancelled"""
        print("Settings cancelled")
        
    def _on_selection_complete(self, result: SelectionResult):
        """Called when BCI selection completes"""
        self.last_selection = result
        
        # End and save session
        if self.session_logger and self.session_logger.is_active:
            filepath = self.session_logger.end_session(
                selected_direction=result.direction
            )
            if filepath:
                print(f"Session saved: {filepath}")
        
        if result.direction:
            print(f"Selection: {result.direction.value} "
                  f"({result.duration_ms:.0f}ms, "
                  f"timing OK: {result.timing_stats['acceptable']})")
            
            # Move player in game
            if self.game_manager and self.game_manager.can_accept_input:
                moved = self.game_manager.move_player(result.direction)
                if not moved:
                    print("  (blocked by wall)")
        else:
            print("Selection: None (timeout or cancelled)")
            
    def _on_flash_start(self, direction: Direction, sequence: int, timestamp_ms: float):
        """Called when a flash begins - log to session"""
        if self.session_logger and self.session_logger.is_active:
            self.session_logger.log_flash_start(direction, sequence, timestamp_ms)
            
    def _on_flash_end(self, direction: Direction, sequence: int, timestamp_ms: float):
        """Called when a flash ends - log to session"""
        if self.session_logger and self.session_logger.is_active:
            self.session_logger.log_flash_end(direction, sequence, timestamp_ms)
            
    def _on_level_complete(self, level: int, score: int):
        """Called when a game level is completed"""
        print(f"Level {level} complete! Score: {score}")
        
    def _on_item_collected(self, points: int):
        """Called when player collects an item"""
        if self.show_debug:
            print(f"  Collected item: +{points} points")
            
    def _on_state_change(self, state: SelectionState):
        """Called when selection state changes"""
        if self.show_debug:
            print(f"  State -> {state.name}")
            
    def _update(self):
        """Update game state"""
        delta_ms = self.clock.get_time()
        
        # Update active stimulus mode
        if self._is_calibration_active():
            self._update_calibration_run()
        else:
            self.arrow_manager.update()
        
        # Update game manager
        if self.game_manager:
            self.game_manager.update(delta_ms)
        
    def _draw(self):
        """Render frame.
        
        All components render to self.render_surface at the design resolution
        (DESIGN_WIDTH x DESIGN_HEIGHT). The result is then scaled to fit the
        actual window, maintaining aspect ratio.
        """
        # Clear render surface
        self.render_surface.fill(self.config.display.background_color)
        
        # Draw game (maze, collectibles, player) - behind arrows
        if self.game_manager:
            self.game_manager.draw(self.render_surface)
        
        # Draw arrows / calibration stimulus (on top of game)
        if self._is_calibration_active():
            self._draw_calibration_stimulus()
        else:
            self.arrow_manager.draw(self.render_surface)
        
        # Draw UI overlays (debug only now - scoreboard is part of game renderer)
        if self.show_debug:
            self._draw_debug()
            
        # Draw settings panel (on top of everything)
        if self.settings_panel:
            self.settings_panel.draw(self.render_surface)
        
        # Scale render surface to actual window
        self.screen.fill((0, 0, 0))  # Black letterbox bars
        if self.scale_factor == 1.0 and self.render_offset == (0, 0):
            # No scaling needed - direct blit
            self.screen.blit(self.render_surface, (0, 0))
        else:
            scaled = pygame.transform.smoothscale(self.render_surface, self.scaled_size)
            self.screen.blit(scaled, self.render_offset)
            
        # Flip display
        pygame.display.flip()
        
    def _draw_status(self):
        """Draw status bar at bottom"""
        # Background bar
        bar_height = 40
        bar_rect = pygame.Rect(
            0, 
            DESIGN_HEIGHT - bar_height,
            DESIGN_WIDTH,
            bar_height
        )
        pygame.draw.rect(self.render_surface, (15, 15, 15), bar_rect)
        
        # Status text
        state = self.arrow_manager.state
        if state == SelectionState.IDLE:
            status = "Press SPACE to start BCI selection"
            color = (50, 50, 50)
        elif state == SelectionState.FLASHING:
            progress = self.arrow_manager.progress * 100
            status = f"Flashing... {progress:.0f}%"
            color = (45, 70, 45)
        elif state == SelectionState.PROCESSING:
            status = "Processing... (Press 1-4 to simulate)"
            color = (70, 70, 45)
        elif state == SelectionState.FEEDBACK:
            status = f"Selected: {self.last_selection.direction.value if self.last_selection else '?'}"
            color = (45, 60, 80)
        else:
            status = str(state.name)
            color = (50, 50, 50)
            
        text = self.font_medium.render(status, True, color)
        text_rect = text.get_rect(
            centerx=DESIGN_WIDTH // 2,
            centery=DESIGN_HEIGHT - bar_height // 2
        )
        self.render_surface.blit(text, text_rect)

    def _draw_calibration_stimulus(self):
        """
        Draw calibration visuals.

        FLASHING stage:
            - Render arrows with one highlighted at a time.
        INSTRUCTION/BREAK stages:
            - Clear stimulus area to a neutral screen with no flashing.
            - Show centered ATTEND instruction during instruction stage.
        """
        if self.calibration_stage == CalibrationStage.FLASHING:
            self.arrow_manager.renderer.draw(self.render_surface, self.calibration_flash_states)
            return

        panel_rect = self.arrow_manager.get_panel_rect()
        if panel_rect:
            neutral_rect = panel_rect.inflate(100, 100)
            pygame.draw.rect(
                self.render_surface,
                self.config.display.background_color,
                neutral_rect,
            )

        center = panel_rect.center if panel_rect else (DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2)

        if (
            self.calibration_stage == CalibrationStage.INSTRUCTION
            and self.calibration_phase_index < len(self.calibration_phase_order)
        ):
            attended = self.calibration_phase_order[self.calibration_phase_index]
            if self._is_warmup_active():
                warmup_num = self.warmup_total - self.warmup_remaining + 1
                header = f"WARM-UP ({warmup_num}/{self.warmup_total})"
                header_surf = self.font_large.render(header, True, (200, 200, 100))
                header_rect = header_surf.get_rect(center=(center[0], center[1] - 30))
                self.render_surface.blit(header_surf, header_rect)

                attend_msg = f"Attend {attended.value.upper()}"
                attend_surf = self.font_medium.render(attend_msg, True, (180, 180, 180))
                attend_rect = attend_surf.get_rect(center=(center[0], center[1] + 20))
                self.render_surface.blit(attend_surf, attend_rect)
            else:
                message = f"ATTEND {attended.value.upper()}"
                text = self.font_large.render(message, True, (200, 200, 200))
                text_rect = text.get_rect(center=center)
                self.render_surface.blit(text, text_rect)

        if self.calibration_stage == CalibrationStage.BREAK and self._is_warmup_active():
            header = "WARM-UP"
            header_surf = self.font_large.render(header, True, (200, 200, 100))
            header_rect = header_surf.get_rect(center=(center[0], center[1] - 15))
            self.render_surface.blit(header_surf, header_rect)

            next_surf = self.font_medium.render("Next trial...", True, (120, 120, 120))
            next_rect = next_surf.get_rect(center=(center[0], center[1] + 25))
            self.render_surface.blit(next_surf, next_rect)

        if self.calibration_stage == CalibrationStage.WAITING:
            wait_message = (
                "Press target arrow"
                if self._is_live_bci_mode()
                else "Press SPACE"
            )
            text = self.font_large.render(wait_message, True, (160, 160, 160))
            text_rect = text.get_rect(center=center)
            self.render_surface.blit(text, text_rect)
        
    def _draw_debug(self):
        """Draw debug information overlay"""
        if self._is_calibration_active():
            state_text = f"CALIBRATION_{self.calibration_stage.name}"
            if self.calibration_stage == CalibrationStage.FLASHING and self.calibration_flash_plan:
                progress = self.calibration_flash_index / len(self.calibration_flash_plan)
            else:
                progress = 0.0
        else:
            state_text = self.arrow_manager.state.name
            progress = self.arrow_manager.progress

        lines = [
            f"FPS: {self.clock.get_fps():.1f}",
            f"State: {state_text}",
            f"Progress: {progress * 100:.0f}%",
            "",
            f"Flash: {self.config.timing.flash_duration_ms}ms",
            f"ISI: {self.config.timing.isi_ms}ms",
            f"Sequences: {self.config.timing.num_sequences}",
            "",
            f"Window: {self.config.display.width}x{self.config.display.height}",
            f"Scale: {self.scale_factor:.2f}x",
        ]

        if self._is_warmup_active() and self.calibration_phase_order:
            warmup_num = self.warmup_total - self.warmup_remaining + 1
            lines.extend([
                "",
                f"WARM-UP: {warmup_num}/{self.warmup_total}",
                f"Attend: {self.calibration_phase_order[self.calibration_phase_index].value.upper()}",
            ])
        elif self._is_calibration_active() and self.calibration_phase_order:
            total_phases = len(self.calibration_phase_order)
            lines.extend([
                "",
                f"Phase: {self.calibration_phase_index + 1}/{total_phases}",
                f"Attend: {self.calibration_phase_order[self.calibration_phase_index].value.upper()}",
            ])
        
        if self.last_selection:
            lines.extend([
                "",
                f"Last: {self.last_selection.direction.value if self.last_selection.direction else 'None'}",
                f"Time: {self.last_selection.duration_ms:.0f}ms",
            ])
            
        # Draw background
        padding = 10
        line_height = 20
        width = 180
        height = len(lines) * line_height + padding * 2
        
        bg_rect = pygame.Rect(
            DESIGN_WIDTH - width - padding,
            padding,
            width,
            height
        )
        bg_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 180))
        self.render_surface.blit(bg_surface, bg_rect.topleft)
        
        # Draw text
        y = padding * 2
        for line in lines:
            if line:
                text = self.font_small.render(line, True, (60, 60, 60))
                self.render_surface.blit(text, 
                    (DESIGN_WIDTH - width, y))
            y += line_height
            
    def _cleanup(self):
        """Clean up resources"""
        if self._is_calibration_active():
            self._finish_calibration_run(cancelled=True)
        elif self.session_logger and self.session_logger.is_active:
            self.session_logger.cancel_session()

        if self.classifier:
            self.classifier.stop()

        if self.arrow_manager:
            self.arrow_manager.shutdown()
        pygame.quit()
        print()
        print("Application closed.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="P300 BCI Game - Salem State University Capstone"
    )
    parser.add_argument(
        "--fullscreen", "-f", 
        action="store_true",
        help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--debug", "-d", 
        action="store_true",
        help="Enable debug overlay"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=3072,
        help="Window width (default: 3072)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=1920,
        help="Window height (default: 1920)"
    )
    parser.add_argument(
        "--sequences", 
        type=int, 
        default=None,
        help="Number of sequences per selection (default: from config)"
    )
    parser.add_argument(
        "--display",
        type=int,
        default=None,
        help="Display index to use (0=primary, 1=secondary, etc.). If not specified, automatically uses second display when available"
    )
    return parser.parse_args()


def main():
    """Entry point"""
    args = parse_args()
    
    # Create configuration
    config = Config()
    config.display.width = args.width
    config.display.height = args.height
    config.display.fullscreen = args.fullscreen
    config.debug = args.debug
    
    if args.sequences:
        config.timing.num_sequences = args.sequences
    
    # Create and run application
    try:
        app = Application(config)
        
        # Allow manual display override via command line
        if args.display is not None:
            app.display_index = args.display
            app.display_override = True
            
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    
    return 0


if __name__ == "__main__":
    sys.exit(main())