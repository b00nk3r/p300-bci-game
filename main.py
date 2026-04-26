#!/usr/bin/env python3
"""
P300 BCI Game - Main Entry Point
================================
Salem State University Capstone Project

A Brain-Computer Interface game using P300 evoked potentials
to control a maze character through flashing arrow stimuli.

The game runs three phases automatically:
    1. CALIBRATION - 10 randomly chosen target directions are flashed and
       EEG / triggers / sessions are recorded so a per-user model can be
       trained.
    2. TRAINING    - the recorded data is fed to
       ``TOBE_INTEGRATED/preprocess_test_epochs.py`` and an LDA model is
       trained, mirroring ``TOBE_INTEGRATED/SingleTrialLDA_10.ipynb``.
       The fitted model is written to ``models/10trials_model.joblib``.
    3. PLAY        - once training finishes, press SPACE to start the
       live BCI game (continuous trials with 10 second pauses between
       selections, as in the ``test_mode`` branch).

Controls:
    SPACE  - Start playing once the model is trained
    D      - Toggle debug overlay
    ESC    - Quit
    Arrows - Manual movement fallback during play
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
import threading
import traceback
from datetime import datetime
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
    PRE_FLASH = auto()     # 3-second gap before flashing begins (BCI mode only)
    FLASHING = auto()
    BREAK = auto()
    WAITING = auto()       # Paused after classification, waiting for next target input


class GamePhase(Enum):
    """Top-level phase the application is currently in."""
    BOOT = auto()           # Just initialized, about to transition.
    CALIBRATING = auto()    # Recording 10 target trials for training data.
    TRAINING = auto()       # Background thread is preprocessing + fitting.
    READY_TO_PLAY = auto()  # Model trained, waiting for SPACE.
    PLAYING = auto()        # Live BCI gameplay, test-mode style auto-loop.


CALIBRATION_TRIALS = 10  # Random target directions to record before training.


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
        self.calibration_break_ms = 10000  # 10-second gap between play trials.
        self.calibration_data_break_ms = 3000  # Shorter gap during data collection.

        # Top-level phase state machine
        self.game_phase = GamePhase.BOOT
        self._calibration_started_at: float = 0.0
        self._calibration_started_dt = None  # datetime, used to filter session files
        self._calibration_completed_count = 0
        self._calibration_target_count = CALIBRATION_TRIALS

        # Training thread state
        self._training_thread: threading.Thread = None
        self._training_done = False
        self._training_error: str = None
        self._training_status_text = "Preparing training data..."

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

        # Auto-start calibration so the player just runs `python main.py`.
        self._start_calibration_phase()
        
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
        """Classifier is loaded lazily after calibration training completes.

        Calibration only needs trigger/session files plus the externally
        recorded HDF5 from ``eeg_acquisition.py`` — a fitted model is not
        required to record data. Once training finishes we call
        :py:meth:`_load_classifier_after_training` to attach the freshly
        trained model to the live game.
        """
        self.classifier = None
        self.arrow_manager.classifier = None

    def _load_classifier_after_training(self) -> bool:
        """Load the freshly trained model and attach it to the arrow manager."""
        from config import BCI_MODE, LSL_STREAM_TYPE, LSL_STREAM_NAME

        if not BCI_MODE:
            print("BCI_MODE is disabled; running gameplay without live classifier.")
            return False

        from realtime_classifier import create_classifier

        try:
            self.classifier = create_classifier(
                stream_type=LSL_STREAM_TYPE,
                stream_name=LSL_STREAM_NAME,
            )
        except Exception as exc:
            print(f"WARNING: Failed to construct classifier: {exc}")
            self.classifier = None
            self.arrow_manager.classifier = None
            return False

        if not self.classifier.start():
            print("WARNING: Could not connect to EEG stream. "
                  "Falling back to keyboard simulation.")
            self.classifier = None
            self.arrow_manager.classifier = None
            return False

        self.arrow_manager.classifier = self.classifier
        return True

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
            max_maze_width=maze_width_cells,
            max_maze_height=maze_height_cells,
            base_collectibles=5,
            max_collectibles=5,
            cell_size=cell_size,
            use_corridors=use_corridors,
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
            on_game_over=self._on_game_over,
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
        print("  Classifier: will be loaded after calibration training")
        print()
        print("Flow:")
        print(f"  1. Calibration: {CALIBRATION_TRIALS} random target trials")
        print("  2. Training:    LDA fitted from collected EEG")
        print("  3. Play:        live BCI gameplay (test-mode style)")
        print()
        print("Controls:")
        print("  SPACE  - Start playing once the model is trained")
        print("  D      - Toggle debug info")
        print("  R      - Restart current game (only after training)")
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
            return

        # SPACE only does something in READY_TO_PLAY (start gameplay) or
        # during PLAYING (resume from waiting / start the next BCI trial).
        if key == pygame.K_SPACE:
            if self.game_phase == GamePhase.READY_TO_PLAY:
                self._enter_play_phase()
                return
            if self.game_phase == GamePhase.PLAYING:
                if self._is_game_over():
                    print("Game already finished — press R to restart.")
                    return
                if self._is_live_bci_mode():
                    if self.calibration_stage in (
                        CalibrationStage.IDLE, CalibrationStage.WAITING
                    ):
                        direction = random.choice(Direction.all())
                        self._start_live_bci_trial(direction)
                    else:
                        print("BCI trial already running")
                elif self.calibration_stage == CalibrationStage.WAITING:
                    self._advance_calibration_phase()
                return
            print(
                f"SPACE ignored — current phase: {self.game_phase.name}"
            )
            return

        # Toggle debug
        if key == pygame.K_d:
            self.show_debug = not self.show_debug
            return

        # Toggle settings panel
        if key == pygame.K_s:
            if self.game_phase == GamePhase.PLAYING:
                self._toggle_settings()
            return

        # Game controls — only after training is done.
        if key == pygame.K_r:
            if self.game_phase == GamePhase.PLAYING and self.game_manager:
                self._finish_calibration_run(cancelled=True)
                self.game_manager.start_game()
                print("Game restarted")
                if self._is_live_bci_mode():
                    direction = random.choice(Direction.all())
                    self._start_live_bci_trial(direction)
            return

        # Simulate selections (for testing) — only during PLAYING.
        if self.game_phase == GamePhase.PLAYING:
            if key == pygame.K_1:
                self._simulate_selection(Direction.UP)
                return
            if key == pygame.K_2:
                self._simulate_selection(Direction.DOWN)
                return
            if key == pygame.K_3:
                self._simulate_selection(Direction.LEFT)
                return
            if key == pygame.K_4:
                self._simulate_selection(Direction.RIGHT)
                return

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
        """Whether a calibration / BCI trial is currently in flight."""
        return self.calibration_stage != CalibrationStage.IDLE

    def _start_calibration_phase(self):
        """Enter the data-collection calibration phase used at game start."""
        self.game_phase = GamePhase.CALIBRATING
        self._calibration_completed_count = 0
        self._calibration_started_at = time.perf_counter()
        # Round to whole seconds so the timestamp matches the trigger filename
        # (which only carries seconds resolution).
        self._calibration_started_dt = datetime.now().replace(microsecond=0)

        # Keep behavior predictable if legacy selection is active.
        if self.arrow_manager.is_active:
            self.arrow_manager.stop_selection()

        # Pick CALIBRATION_TRIALS random target directions (with repetition).
        self.calibration_phase_order = [
            random.choice(Direction.all()) for _ in range(self._calibration_target_count)
        ]
        self.calibration_phase_index = 0
        self.calibration_run_start_time = time.perf_counter()

        self._begin_single_calibration_trial(
            self.calibration_phase_order[self.calibration_phase_index]
        )
        print(
            f"Calibration phase started — {self._calibration_target_count} "
            f"random-target trials"
        )
        print(f"  Targets: {[d.value for d in self.calibration_phase_order]}")

    def _begin_single_calibration_trial(self, target: Direction):
        """Start a single labeled calibration trial (one session/trigger pair)."""
        # Close any session that was left open by an earlier trial.
        if self.session_logger and self.session_logger.is_active:
            self.session_logger.end_session()
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

        self.calibration_run_start_time = time.perf_counter()
        self._set_calibration_idle_arrows()
        self._start_instruction_stage()

    def _start_live_bci_trial(self, target: Direction):
        """Start one labeled BCI gameplay trial for the given target arrow."""
        if not self._is_live_bci_mode():
            return

        if self._is_calibration_active() and self.calibration_stage not in (
            CalibrationStage.IDLE,
            CalibrationStage.WAITING,
            CalibrationStage.BREAK,
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
        """Show 'ATTEND <DIR>' before flashing during calibration; skip in play."""
        attended = self.calibration_phase_order[self.calibration_phase_index]
        self.arrow_manager.triggers.set_current_target(attended)
        total_phases = len(self.calibration_phase_order)

        if self.game_phase == GamePhase.CALIBRATING:
            self.calibration_stage = CalibrationStage.INSTRUCTION
            self.calibration_stage_start_time = time.perf_counter()
            self._set_calibration_idle_arrows()
            print(
                f"Phase {self.calibration_phase_index + 1}/{total_phases} - "
                f"ATTEND {attended.value.upper()}"
            )
            return

        print(
            f"Phase {self.calibration_phase_index + 1}/{total_phases} - "
            f"Flashing {attended.value.upper()}"
        )
        self._start_flashing_stage()

    def _start_pre_flash_stage(self):
        # 3 second delay
        self.calibration_stage = CalibrationStage.PRE_FLASH
        self.calibration_stage_start_time = time.perf_counter()
        self._set_calibration_idle_arrows()

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
        """Advance to the next trial in the current phase or finish it."""
        if self.game_phase == GamePhase.CALIBRATING:
            self._advance_data_collection_trial()
            return

        # Live BCI gameplay — auto-loop with random targets.
        if self._is_live_bci_mode():
            direction = random.choice(Direction.all())
            self._start_live_bci_trial(direction)

    def _advance_data_collection_trial(self):
        """Move to the next data-collection trial or trigger training."""
        # The trial that just finished gets persisted to disk.
        self.arrow_manager.triggers.send_trial_end()
        self.arrow_manager.triggers.stop_session()
        if self.session_logger and self.session_logger.is_active:
            self.session_logger.end_session()

        self._calibration_completed_count += 1
        self.calibration_phase_index += 1

        remaining = self._calibration_target_count - self._calibration_completed_count
        print(
            f"Calibration trial {self._calibration_completed_count}/"
            f"{self._calibration_target_count} complete ({remaining} remaining)"
        )

        if self.calibration_phase_index >= len(self.calibration_phase_order):
            self._enter_training_phase()
            return

        next_target = self.calibration_phase_order[self.calibration_phase_index]
        self._begin_single_calibration_trial(next_target)

    def _enter_training_phase(self):
        """Move from CALIBRATING to TRAINING and kick off the pipeline thread."""
        self.calibration_stage = CalibrationStage.IDLE
        self.calibration_phase_order = []
        self.calibration_phase_index = 0
        self.calibration_flash_plan = []
        self.calibration_flash_index = 0
        self._set_calibration_idle_arrows()
        self.arrow_manager.triggers.set_current_target(None)

        self.game_phase = GamePhase.TRAINING
        self._training_done = False
        self._training_error = None
        self._training_status_text = "Preprocessing calibration data..."
        print(
            f"All {self._calibration_target_count} calibration trials recorded."
            " Starting training..."
        )

        self._training_thread = threading.Thread(
            target=self._run_training_pipeline,
            name="calibration-training",
            daemon=True,
        )
        self._training_thread.start()

    def _run_training_pipeline(self):
        """Background-thread worker that runs preprocess + train."""
        try:
            from train_calibration_pipeline import run_pipeline

            self._training_status_text = "Preprocessing calibration data..."
            run_pipeline(
                n_runs=self._calibration_target_count,
                after=self._calibration_started_dt,
            )
        except Exception as exc:  # noqa: BLE001
            self._training_error = f"{exc}"
            traceback.print_exc()
            return

        self._training_status_text = "Model trained! Press SPACE to play."
        self._training_done = True

    def _enter_ready_to_play_phase(self):
        """Transition from TRAINING -> READY_TO_PLAY and load the new model."""
        self.game_phase = GamePhase.READY_TO_PLAY
        loaded = self._load_classifier_after_training()
        if not loaded:
            self._training_status_text = (
                "Model trained but classifier could not connect. "
                "Press SPACE to start gameplay anyway."
            )
        else:
            self._training_status_text = "Model ready! Press SPACE to play."

    def _enter_play_phase(self):
        """Transition into live BCI gameplay (test-mode behavior)."""
        if self.game_phase != GamePhase.READY_TO_PLAY:
            return
        self.game_phase = GamePhase.PLAYING

        if self.game_manager:
            self.game_manager.start_game()

        if self._is_live_bci_mode():
            direction = random.choice(Direction.all())
            self._start_live_bci_trial(direction)
            print("Gameplay started — first trial running.")
        else:
            print("Gameplay started — no live classifier, manual play only.")

    def _finish_calibration_run(self, cancelled: bool = False):
        """Cancel an in-flight calibration trial without entering training."""
        was_active = self._is_calibration_active()

        self.calibration_stage = CalibrationStage.IDLE
        self.calibration_phase_order = []
        self.calibration_phase_index = 0
        self.calibration_stage_start_time = 0.0
        self.calibration_flash_plan = []
        self.calibration_flash_index = 0
        self.calibration_next_flash_time = 0.0
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

    def _update_calibration_run(self):
        """Advance the calibration run state machine."""
        now = time.perf_counter()
        elapsed_stage_ms = (now - self.calibration_stage_start_time) * 1000.0

        if self.calibration_stage == CalibrationStage.INSTRUCTION:
            if elapsed_stage_ms >= self.calibration_instruction_ms:
                self._start_flashing_stage()
            return

        if self.calibration_stage == CalibrationStage.PRE_FLASH:
            if elapsed_stage_ms >= 3000.0:
                self._start_flashing_stage()
            return

        if self.calibration_stage == CalibrationStage.FLASHING:
            self._update_calibration_flashing(now)
            return

        if self.calibration_stage == CalibrationStage.WAITING:
            return

        if self.calibration_stage == CalibrationStage.BREAK:
            break_ms = (
                self.calibration_data_break_ms
                if self.game_phase == GamePhase.CALIBRATING
                else self.calibration_break_ms
            )
            if elapsed_stage_ms >= break_ms:
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

                if self._is_live_bci_mode():
                    self._finalize_live_bci_trial(selected_direction)
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
            
    def _is_game_over(self) -> bool:
        """Return True when the game manager has reached the GAME_OVER state."""
        return (
            self.game_manager is not None
            and self.game_manager._state == GameState.GAME_OVER
        )

    def _on_game_over(self, stats):
        """Called when the game is finished - stop BCI and display win screen."""
        print(f"Game finished! Score: {stats.score}")
        # Stop any active BCI trial / calibration run so arrows freeze.
        self._finish_calibration_run(cancelled=False)
        
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

        # Watch for training completion and advance the phase machine.
        if self.game_phase == GamePhase.TRAINING and self._training_done:
            self._enter_ready_to_play_phase()

        # Freeze BCI/arrows once the game is won.
        if not self._is_game_over():
            if self._is_calibration_active():
                self._update_calibration_run()
            elif self.game_phase == GamePhase.PLAYING:
                self.arrow_manager.update()

        # Only update the game (movement, animations) once we're playing.
        if self.game_phase == GamePhase.PLAYING and self.game_manager:
            self.game_manager.update(delta_ms)
        
    def _draw(self):
        """Render frame.
        
        All components render to self.render_surface at the design resolution
        (DESIGN_WIDTH x DESIGN_HEIGHT). The result is then scaled to fit the
        actual window, maintaining aspect ratio.
        """
        # Clear render surface
        self.render_surface.fill(self.config.display.background_color)

        # Game world is only visible during PLAYING (and on the win screen).
        if self.game_manager and self.game_phase == GamePhase.PLAYING:
            self.game_manager.draw(self.render_surface)

        # Draw arrows / calibration stimulus on top of the game.
        if self.game_phase == GamePhase.CALIBRATING:
            self._draw_calibration_stimulus()
            self._draw_calibration_overlay()
        elif self.game_phase == GamePhase.TRAINING:
            self._draw_training_overlay()
        elif self.game_phase == GamePhase.READY_TO_PLAY:
            self._draw_ready_overlay()
        elif self.game_phase == GamePhase.PLAYING and not self._is_game_over():
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
            message = f"ATTEND {attended.value.upper()}"
            text = self.font_large.render(message, True, (200, 200, 200))
            text_rect = text.get_rect(center=center)
            self.render_surface.blit(text, text_rect)

        if self.calibration_stage == CalibrationStage.BREAK:
            elapsed_ms = (time.perf_counter() - self.calibration_stage_start_time) * 1000.0
            remaining_ms = self.calibration_break_ms - elapsed_ms
            if remaining_ms <= 3000:
                text = self.font_large.render("Get ready...", True, (130, 130, 130))
                text_rect = text.get_rect(center=center)
                self.render_surface.blit(text, text_rect)

        if self.calibration_stage == CalibrationStage.WAITING:
            wait_message = (
                "Press target arrow"
                if self._is_live_bci_mode()
                else "Press SPACE"
            )
            text = self.font_large.render(wait_message, True, (160, 160, 160))
            text_rect = text.get_rect(center=center)
            self.render_surface.blit(text, text_rect)

    def _draw_calibration_overlay(self):
        """Draw progress text for the data-collection calibration phase."""
        progress = (
            f"Calibration trial "
            f"{self._calibration_completed_count + 1}/"
            f"{self._calibration_target_count}"
        )
        progress_surface = self.font_medium.render(progress, True, (180, 180, 180))
        progress_rect = progress_surface.get_rect(
            center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT - 120)
        )
        self.render_surface.blit(progress_surface, progress_rect)

        hint_surface = self.font_small.render(
            "Look at the arrow named on screen — keep your gaze on it.",
            True, (130, 130, 130),
        )
        hint_rect = hint_surface.get_rect(
            center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT - 80)
        )
        self.render_surface.blit(hint_surface, hint_rect)

    def _draw_training_overlay(self):
        """Show the training-progress message centered on screen."""
        title = self.font_large.render("Training model...", True, (220, 220, 220))
        title_rect = title.get_rect(center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2 - 40))
        self.render_surface.blit(title, title_rect)

        if self._training_error:
            err_text = self.font_medium.render(
                f"Error: {self._training_error}", True, (220, 80, 80)
            )
            err_rect = err_text.get_rect(
                center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2 + 30)
            )
            self.render_surface.blit(err_text, err_rect)
            return

        status = self.font_medium.render(
            self._training_status_text, True, (180, 180, 180),
        )
        status_rect = status.get_rect(
            center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2 + 30)
        )
        self.render_surface.blit(status, status_rect)

    def _draw_ready_overlay(self):
        """Show the “press SPACE to play” prompt."""
        title = self.font_large.render(
            "Game ready to be played", True, (220, 220, 220),
        )
        title_rect = title.get_rect(
            center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2 - 40)
        )
        self.render_surface.blit(title, title_rect)

        prompt = self.font_medium.render(
            "Press SPACE to start", True, (180, 220, 180),
        )
        prompt_rect = prompt.get_rect(
            center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2 + 30)
        )
        self.render_surface.blit(prompt, prompt_rect)

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

        if self._is_calibration_active() and self.calibration_phase_order:
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