import sys
import argparse
import time
import random
from enum import Enum, auto

import pygame

from config import Config, Direction
from src.stimulus.stimulus_controller import StimulusController
from src.ui.settings_panel import SettingsPanel, SettingsValues
from src.game.game_manager import GameManager, GameManagerConfig
from src.data.session_logger import SessionLogger

DESIGN_WIDTH = 3072
DESIGN_HEIGHT = 1920


class CalibrationStage(Enum):
    IDLE = auto()
    INSTRUCTION = auto()
    FLASHING = auto()
    BREAK = auto()


class Application:

    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.clock = None
        self.screen = None

        self.stim: StimulusController = None
        self.settings_panel: SettingsPanel = None
        self.game_manager: GameManager = None
        self.session_logger: SessionLogger = None

        self.render_surface = None
        self.scale_factor = 1.0
        self.render_offset = (0, 0)
        self.scaled_size = (DESIGN_WIDTH, DESIGN_HEIGHT)

        self.num_displays = 1
        self.display_index = 0
        self.display_override = False

        self.font_large = None
        self.font_medium = None
        self.font_small = None

        # Calibration run state machine. The flashing itself is handled by the
        # StimulusController; this class only drives the phase / instruction /
        # break sequence around it.
        self.calibration_stage = CalibrationStage.IDLE
        self.calibration_phase_order = []
        self.calibration_phase_index = 0
        self.calibration_stage_start_time = 0.0
        self.calibration_run_start_time = 0.0

        self.calibration_instruction_ms = 2000
        self.calibration_break_ms = 2000

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

        if self.config.display.fullscreen:
            display_info = pygame.display.Info()
            self.config.display.width = display_info.current_w
            self.config.display.height = display_info.current_h

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

        # Initialize the stimulus system (renderer + timing + triggers)
        self._init_stimulus()

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

    def _init_stimulus(self):
        """Initialize the stimulus controller and register flash callbacks."""
        self.stim = StimulusController(self.config)
        self.stim.initialize(DESIGN_WIDTH, DESIGN_HEIGHT)

        self.stim.set_callbacks(
            on_flash_start=self._on_flash_start,
            on_flash_end=self._on_flash_end,
            on_block_complete=self._on_block_complete,
        )

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
        panel_rect = self.stim.get_panel_rect()

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
            base_collectibles=2,  # Fixed donut count per level
            collectibles_per_level=0,
            max_collectibles=2,
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
        print("Controls:")
        print("  S      - Open settings panel")
        print("  R      - Restart current level")
        print("  N      - Skip to next level")
        print("  ESC    - Quit")
        print("  SPACE  - Start calibration run")
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

    def _handle_keydown(self, key: int):
        if key == pygame.K_ESCAPE:
            self.running = False

        elif key == pygame.K_SPACE:
            if self.calibration_stage == CalibrationStage.IDLE:
                self._start_calibration_run()
            elif self._is_calibration_active():
                print("Calibration already running")

        elif key == pygame.K_s:
            self._toggle_settings()

        elif key == pygame.K_r:
            if self._is_calibration_active():
                print("Cannot restart during an active trial")
                return
            elif self.game_manager:
                self.game_manager.restart_level()
                print("Level restarted")
        elif key == pygame.K_n:
            if self._is_calibration_active():
                print("Cannot skip levels during an active trial")
                return
            if self.game_manager:
                self.game_manager.next_level()
                print(f"Advanced to level {self.game_manager.stats.level}")

    def _is_calibration_active(self) -> bool:
        """Whether a calibration run is currently active."""
        return self.calibration_stage != CalibrationStage.IDLE

    def _start_calibration_run(self):
        """Start one full data-recording calibration run."""
        if self._is_calibration_active():
            print("Calibration already running")
            return

        self.calibration_phase_order = []
        for _ in range(3):
            block = Direction.all()
            random.shuffle(block)
            self.calibration_phase_order.extend(block)

        self.calibration_phase_index = 0
        self.calibration_run_start_time = time.perf_counter()

        if self.session_logger:
            self.session_logger.start_session(
                flash_duration_ms=self.config.timing.flash_duration_ms,
                isi_ms=self.config.timing.isi_ms,
                num_sequences=self.config.timing.num_sequences,
                inter_sequence_pause_ms=0,
                flash_pattern="RANDOM",
                color_scheme=self.config.arrows.color_scheme.name,
            )

        # One trigger session for the whole run (all 12 phases).
        self.stim.triggers.start_session(
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
        self.stim.triggers.set_current_target(first_target)
        self.stim.triggers.send_trial_start()

        self._start_instruction_stage()
        print("Calibration run started")

    def _start_instruction_stage(self):
        attended = self.calibration_phase_order[self.calibration_phase_index]
        self.stim.triggers.set_current_target(attended)
        total_phases = len(self.calibration_phase_order)

        self.calibration_stage = CalibrationStage.INSTRUCTION
        self.calibration_stage_start_time = time.perf_counter()
        print(
            f"Phase {self.calibration_phase_index + 1}/{total_phases} - "
            f"ATTEND {attended.value.upper()}"
        )

    def _start_flashing_stage(self):
        """Begin one flash block for the current attended-arrow phase."""
        attended = self.calibration_phase_order[self.calibration_phase_index]

        self.calibration_stage = CalibrationStage.FLASHING
        self.calibration_stage_start_time = time.perf_counter()

        # The controller flashes num_sequences sequences of the four arrows,
        # using config.timing for duration/ISI, and reports each flash and the
        # block completion through the callbacks registered in _init_stimulus.
        self.stim.start_block(target=attended)

    def _start_break_stage(self):
        """Start blank/neutral break between attended-arrow phases."""
        self.calibration_stage = CalibrationStage.BREAK
        self.calibration_stage_start_time = time.perf_counter()
        self.stim.triggers.set_current_target(None)

    def _advance_calibration_phase(self):
        self.calibration_phase_index += 1
        if self.calibration_phase_index >= len(self.calibration_phase_order):
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

        if was_active:
            self.stim.triggers.set_current_target(None)
            self.stim.triggers.send_trial_end()
            self.stim.triggers.stop_session()

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
        now = time.perf_counter()
        elapsed_stage_ms = (now - self.calibration_stage_start_time) * 1000.0

        if self.calibration_stage == CalibrationStage.INSTRUCTION:
            if elapsed_stage_ms >= self.calibration_instruction_ms:
                self._start_flashing_stage()
            return

        if self.calibration_stage == CalibrationStage.FLASHING:
            # Advance the flash engine. When the block finishes, the controller
            # calls _on_block_complete, which moves us to the break stage.
            self.stim.update()
            return

        if self.calibration_stage == CalibrationStage.BREAK:
            if elapsed_stage_ms >= self.calibration_break_ms:
                self._advance_calibration_phase()

    def _toggle_settings(self):
        """Toggle the settings panel"""
        if self.settings_panel.is_visible:
            self.settings_panel.hide()
        else:
            # Don't open settings during an active calibration run
            if self._is_calibration_active():
                print("Cannot open settings during an active run")
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

        # Rebuild the arrow surfaces for the new appearance. Timing values are
        # read live from config by the controller, so no other rebuild is needed.
        self.stim.renderer.set_color_scheme(self.config.arrows.color_scheme)

        # Update game manager dullness
        if self.game_manager:
            self.game_manager.set_dullness(self.config.game.dullness)

    def _on_settings_cancel(self):
        """Called when settings are cancelled"""
        print("Settings cancelled")

    def _on_flash_start(self, direction: Direction, sequence: int, time_ms: float):
        """Called by the controller when a flash begins - log to session.

        `time_ms` from the controller is relative to the current block. We log a
        run-relative timestamp instead, so the session log matches the original
        (timestamps measured from the start of the whole run).
        """
        if self.session_logger and self.session_logger.is_active:
            timestamp_ms = (time.perf_counter() - self.calibration_run_start_time) * 1000.0
            self.session_logger.log_flash_start(direction, sequence, timestamp_ms)

    def _on_flash_end(self, direction: Direction, sequence: int, time_ms: float):
        """Called by the controller when a flash ends - log to session."""
        if self.session_logger and self.session_logger.is_active:
            timestamp_ms = (time.perf_counter() - self.calibration_run_start_time) * 1000.0
            self.session_logger.log_flash_end(direction, sequence, timestamp_ms)

    def _on_block_complete(self):
        """Called by the controller when a phase's flash block finishes."""
        self._start_break_stage()

    def _on_level_complete(self, level: int, score: int):
        """Called when a game level is completed"""
        print(f"Level {level} complete! Score: {score}")

    def _on_item_collected(self, points: int):
        pass

    def _update(self):
        """Update game state"""
        delta_ms = self.clock.get_time()

        # Drive the calibration run when one is active.
        if self._is_calibration_active():
            self._update_calibration_run()

        # Update game manager
        if self.game_manager:
            self.game_manager.update(delta_ms)

    def _draw(self):
        self.render_surface.fill(self.config.display.background_color)

        if self.game_manager:
            self.game_manager.draw(self.render_surface)

        # The controller draws the panel plus the arrows in their current flash
        # states (all idle when not flashing), in every stage.
        self.stim.draw(self.render_surface)

        # Overlay the ATTEND instruction during the instruction stage.
        if (
            self.calibration_stage == CalibrationStage.INSTRUCTION
            and self.calibration_phase_index < len(self.calibration_phase_order)
        ):
            self._draw_attend_overlay()

        if self.settings_panel:
            self.settings_panel.draw(self.render_surface)

        self.screen.fill((0, 0, 0))
        if self.scale_factor == 1.0 and self.render_offset == (0, 0):
            self.screen.blit(self.render_surface, (0, 0))
        else:
            scaled = pygame.transform.smoothscale(self.render_surface, self.scaled_size)
            self.screen.blit(scaled, self.render_offset)

        pygame.display.flip()

    def _draw_attend_overlay(self):
        panel_rect = self.stim.get_panel_rect()
        center = panel_rect.center if panel_rect else (DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2)
        attended = self.calibration_phase_order[self.calibration_phase_index]
        message = f"ATTEND {attended.value.upper()}"
        text = self.font_large.render(message, True, (200, 200, 200))
        text_rect = text.get_rect(center=center)
        self.render_surface.blit(text, text_rect)

    def _cleanup(self):
        """Clean up resources"""
        if self._is_calibration_active():
            self._finish_calibration_run(cancelled=True)
        elif self.session_logger and self.session_logger.is_active:
            self.session_logger.cancel_session()

        if self.stim:
            self.stim.shutdown()
        pygame.quit()
        print()
        print("Application closed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P300 BCI Game - Salem State University Capstone"
    )
    parser.add_argument(
        "--fullscreen", "-f",
        action="store_true",
        help="Run in fullscreen mode"
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
    args = parse_args()

    config = Config()
    config.display.width = args.width
    config.display.height = args.height
    config.display.fullscreen = args.fullscreen

    # The original calibration flashed with no pause between sequences (its
    # hand-rolled loop ignored inter_sequence_pause_ms). The timing engine
    # honors this value, so set it to 0 to keep the flash timing identical.
    # Remove this line if you want the configured inter-sequence pause instead.
    config.timing.inter_sequence_pause_ms = 0

    if args.sequences:
        config.timing.num_sequences = args.sequences

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