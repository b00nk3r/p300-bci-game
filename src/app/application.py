import argparse
import time

import pygame

from config import Config
from src.stimulus.stimulus_controller import StimulusController
from src.ui.settings_panel import SettingsPanel, SettingsValues
from src.game.game_manager import GameManager
from src.data.session_logger import SessionLogger

DESIGN_WIDTH = 3072
DESIGN_HEIGHT = 1920


class Application:
    """Shared skeleton for the P300 BCI app.

    Owns everything the calibration and live modes have in common: the window,
    resolution-independent scaling and letterboxing, fonts, the stimulus
    controller, the settings panel, the maze game, the event loop, settings
    handling, flash logging, and cleanup.

    Mode-specific behavior lives in subclasses (CalibrationMode, LiveMode),
    which override the hook methods grouped at the bottom of this class. The
    base never instantiates a mode itself; run_main() picks the subclass.
    """

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

        # Origin for run-relative flash timestamps in the session log. Each mode
        # sets this when it starts a run / selection.
        self.session_log_origin = 0.0

    # ===================================================================
    # Setup
    # ===================================================================

    def initialize(self):
        pygame.init()

        # Detect available displays
        self.num_displays = pygame.display.get_num_displays()

        # Auto-select display if not manually overridden
        if not self.display_override:
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
            display=self.display_index,
        )
        pygame.display.set_caption("P300 BCI Game - Salem State University")

        self.clock = pygame.time.Clock()

        # Render surface at design resolution
        self.render_surface = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))
        self._calculate_scaling()

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Components
        self._init_stimulus()
        self._init_extra()          # mode hook (e.g. classifier); default no-op
        self._init_session_logger()
        self._init_settings_panel()
        self._init_game_manager()
        self._print_startup_info()

    def _calculate_scaling(self):
        """Map the design resolution to the window, letterboxing if needed."""
        window_w = self.config.display.width
        window_h = self.config.display.height

        scale_x = window_w / DESIGN_WIDTH
        scale_y = window_h / DESIGN_HEIGHT
        self.scale_factor = min(scale_x, scale_y)

        scaled_w = int(DESIGN_WIDTH * self.scale_factor)
        scaled_h = int(DESIGN_HEIGHT * self.scale_factor)
        self.scaled_size = (scaled_w, scaled_h)
        self.render_offset = (
            (window_w - scaled_w) // 2,
            (window_h - scaled_h) // 2,
        )

    def _transform_event(self, event):
        """Transform mouse coordinates from window space to design space."""
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            x, y = event.pos
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
        self.stim = StimulusController(self.config)
        self.stim.initialize(DESIGN_WIDTH, DESIGN_HEIGHT)
        self.stim.set_callbacks(
            on_flash_start=self._on_flash_start,
            on_flash_end=self._on_flash_end,
            on_block_complete=self._on_block_complete,
        )

    def _init_session_logger(self):
        self.session_logger = SessionLogger(output_dir="data/sessions")

    def _init_settings_panel(self):
        self.settings_panel = SettingsPanel(DESIGN_WIDTH, DESIGN_HEIGHT)
        values = SettingsValues.from_config(self.config)
        self.settings_panel.set_values(values)
        self.settings_panel.set_callbacks(
            on_apply=self._on_settings_apply,
            on_cancel=self._on_settings_cancel,
        )

    def _init_game_manager(self):
        panel_rect = self.stim.get_panel_rect()

        target_width_cells = 10
        target_height_cells = 6
        cell_size = min(
            DESIGN_WIDTH // target_width_cells,
            DESIGN_HEIGHT // target_height_cells,
        )
        use_corridors = cell_size >= 128

        game_config = self._build_game_config(
            maze_width_cells=target_width_cells,
            maze_height_cells=target_height_cells,
            cell_size=cell_size,
            use_corridors=use_corridors,
        )
        self.game_manager = GameManager(game_config)
        self.game_manager.initialize(DESIGN_WIDTH, DESIGN_HEIGHT, panel_rect)

        self._register_game_callbacks()
        self.game_manager.set_dullness(self.config.game.dullness)
        self.game_manager.start_game()

    def _print_startup_info(self):
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
        self._print_mode_info()
        print("=" * 50)
        print()

    # ===================================================================
    # Main loop
    # ===================================================================

    def run(self):
        self.running = True
        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(self.config.display.fps)
        self._cleanup()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            event = self._transform_event(event)

            if self.settings_panel and self.settings_panel.is_visible:
                if self.settings_panel.handle_event(event):
                    continue  # Event consumed by panel

            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)

    def _handle_keydown(self, key: int):
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_s:
            self._toggle_settings()
        else:
            self._handle_mode_key(key)

    def _toggle_settings(self):
        if self.settings_panel.is_visible:
            self.settings_panel.hide()
        else:
            if self._is_active():
                print("Cannot open settings during an active run")
                return
            values = SettingsValues.from_config(self.config)
            self.settings_panel.set_values(values)
            self.settings_panel.show()
            print("Settings panel opened")

    def _on_settings_apply(self, values: SettingsValues):
        print("Settings applied:")
        print(f"  Flash: {values.flash_duration_ms}ms")
        print(f"  ISI: {values.isi_ms}ms")
        print(f"  Sequences: {values.num_sequences}")
        print(f"  Dullness: {values.dullness}")
        print(f"  Color: {values.color_scheme.name}")
        print(f"  SOA: {values.soa_ms}ms ({values.flash_rate_hz:.1f}Hz)")

        values.apply_to_config(self.config)

        # Rebuild arrow surfaces for the new appearance. Timing is read live by
        # the controller, and the classifier (if any) stays attached, so nothing
        # else needs rebuilding.
        self.stim.renderer.set_color_scheme(self.config.arrows.color_scheme)

        if self.game_manager:
            self.game_manager.set_dullness(self.config.game.dullness)

    def _on_settings_cancel(self):
        print("Settings cancelled")

    def _on_flash_start(self, direction, sequence, time_ms):
        """Controller callback: a flash began. Log it run-relative.

        `time_ms` from the controller is relative to the current block; we log a
        timestamp relative to the run / selection start (session_log_origin) to
        match the original behavior.
        """
        if self.session_logger and self.session_logger.is_active:
            timestamp_ms = (time.perf_counter() - self.session_log_origin) * 1000.0
            self.session_logger.log_flash_start(direction, sequence, timestamp_ms)

    def _on_flash_end(self, direction, sequence, time_ms):
        """Controller callback: a flash ended. Log it run-relative."""
        if self.session_logger and self.session_logger.is_active:
            timestamp_ms = (time.perf_counter() - self.session_log_origin) * 1000.0
            self.session_logger.log_flash_end(direction, sequence, timestamp_ms)

    def _update(self):
        delta_ms = self.clock.get_time()
        self._update_mode()
        if self.game_manager:
            self.game_manager.update(delta_ms)

    def _draw(self):
        self.render_surface.fill(self.config.display.background_color)

        if self.game_manager:
            self.game_manager.draw(self.render_surface)

        self._draw_mode(self.render_surface)

        if self.settings_panel:
            self.settings_panel.draw(self.render_surface)

        self.screen.fill((0, 0, 0))
        if self.scale_factor == 1.0 and self.render_offset == (0, 0):
            self.screen.blit(self.render_surface, (0, 0))
        else:
            scaled = pygame.transform.smoothscale(self.render_surface, self.scaled_size)
            self.screen.blit(scaled, self.render_offset)

        pygame.display.flip()

    def _cleanup(self):
        self._on_cleanup()
        if self.stim:
            self.stim.shutdown()
        pygame.quit()
        print()
        print("Application closed.")

    # ===================================================================
    # Mode hooks - implemented by subclasses
    # ===================================================================

    def _init_extra(self):
        """Optional extra setup after the stimulus controller (e.g. classifier)."""
        pass

    def _on_item_collected(self, points: int):
        """Game callback for collecting an item. Default: nothing."""
        pass

    def _build_game_config(self, maze_width_cells, maze_height_cells, cell_size, use_corridors):
        """Return the GameManagerConfig for this mode."""
        raise NotImplementedError

    def _register_game_callbacks(self):
        """Register this mode's game-manager callbacks."""
        raise NotImplementedError

    def _print_mode_info(self):
        """Print the mode-specific startup section (controls, classifier, etc.)."""
        raise NotImplementedError

    def _handle_mode_key(self, key: int):
        """Handle a key press that is not ESC or S."""
        raise NotImplementedError

    def _is_active(self) -> bool:
        """Whether a run / trial is currently active."""
        raise NotImplementedError

    def _update_mode(self):
        """Advance the mode's state machine (called once per frame)."""
        raise NotImplementedError

    def _draw_mode(self, surface):
        """Draw the mode's stimulus and overlays onto the design surface."""
        raise NotImplementedError

    def _on_block_complete(self):
        """Controller callback: a flash block finished."""
        raise NotImplementedError

    def _on_cleanup(self):
        """Finalize the mode on shutdown (cancel run / trial, stop devices)."""
        raise NotImplementedError


def run_main(mode_class, default_width: int, default_height: int) -> int:
    """Parse args, build config, and run the given mode. Used by the launchers."""
    parser = argparse.ArgumentParser(
        description="P300 BCI Game - Salem State University Capstone"
    )
    parser.add_argument(
        "--fullscreen", "-f",
        action="store_true",
        help="Run in fullscreen mode",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=default_width,
        help=f"Window width (default: {default_width})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=default_height,
        help=f"Window height (default: {default_height})",
    )
    parser.add_argument(
        "--sequences",
        type=int,
        default=None,
        help="Number of sequences per selection (default: from config)",
    )
    parser.add_argument(
        "--display",
        type=int,
        default=None,
        help="Display index to use (0=primary, 1=secondary, etc.). If not specified, automatically uses second display when available",
    )
    args = parser.parse_args()

    config = Config()
    config.display.width = args.width
    config.display.height = args.height
    config.display.fullscreen = args.fullscreen

    # Both modes flash with no pause between sequences, matching the original
    # hand-rolled loops. Remove this line to honor config.timing.inter_sequence_pause_ms.
    config.timing.inter_sequence_pause_ms = 0

    if args.sequences:
        config.timing.num_sequences = args.sequences

    try:
        app = mode_class(config)

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
