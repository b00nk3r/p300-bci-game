import sys
import argparse
import time
import random
from enum import Enum, auto

import pygame

from config import Config, Direction
from src.stimulus.stimulus_controller import StimulusController
from src.ui.settings_panel import SettingsPanel, SettingsValues
from src.game.game_manager import GameManager, GameManagerConfig, GameState
from src.data.session_logger import SessionLogger


DESIGN_WIDTH = 3072
DESIGN_HEIGHT = 1920


class TrialStage(Enum):
    IDLE = auto()
    FLASHING = auto()
    BREAK = auto()


class MockClassifier:

    def __init__(self):
        self._events = []

    def start(self):
        print("MockClassifier started (no EEG device)")
        return True

    def stop(self):
        pass

    def clear_events(self):
        self._events = []

    def record_flash(self, direction, timestamp):
        self._events.append({"direction": direction, "timestamp": timestamp})

    def classify_trial(self):
        if not self._events:
            return None
        # Pick a random direction from the ones that were flashed
        directions = list(set(e["direction"] for e in self._events))
        chosen = random.choice(directions)
        return {
            "direction": chosen,
            "confidence": round(random.uniform(0.4, 0.95), 3),
            "n_epochs_used": len(self._events),
            "n_epochs_total": len(self._events),
        }


class Application:

    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.clock = None
        self.screen = None

        # Components
        self.stim: StimulusController = None
        self.settings_panel: SettingsPanel = None
        self.game_manager: GameManager = None
        self.session_logger: SessionLogger = None
        self.classifier = None

        # Resolution-independent rendering
        self.render_surface = None
        self.scale_factor = 1.0
        self.render_offset = (0, 0)
        self.scaled_size = (DESIGN_WIDTH, DESIGN_HEIGHT)

        # Display info
        self.num_displays = 1
        self.display_index = 0
        self.display_override = False

        # Fonts
        self.font_large = None
        self.font_medium = None
        self.font_small = None

        # Live trial state machine. The flashing is handled by the
        # StimulusController; this class drives the selection lifecycle around
        # it (start, classify, move, break, loop).
        self.trial_stage = TrialStage.IDLE
        self.trial_stage_start_time = 0.0
        self.trial_run_start_time = 0.0

        self.trial_break_ms = 10000  # 10-second gap between selections

    def initialize(self):
        pygame.init()

        # Detect available displays
        self.num_displays = pygame.display.get_num_displays()

        # Auto-select display if not manually overridden
        if not self.display_override:
            # Use second display if available, otherwise use primary
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

        self.render_surface = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))
        self._calculate_scaling()

        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        self._init_stimulus()

        self._init_classifier()

        self._init_session_logger()

        self._init_settings_panel()

        self._init_game_manager()

        self._print_startup_info()

    def _calculate_scaling(self):
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
            (window_h - scaled_h) // 2
        )

    def _transform_event(self, event):
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

    def _init_classifier(self):
        from config import LSL_STREAM_TYPE, LSL_STREAM_NAME

        self.classifier = None
        try:
            from realtime_classifier import create_classifier
            self.classifier = create_classifier(
                stream_type=LSL_STREAM_TYPE,
                stream_name=LSL_STREAM_NAME,
            )
            if not self.classifier.start():
                print("WARNING: Could not connect to EEG stream. Falling back to mock classifier.")
                self.classifier = MockClassifier()
                self.classifier.start()
        except Exception as e:
            print(f"WARNING: Classifier init failed ({e}). Falling back to mock classifier.")
            self.classifier = MockClassifier()
            self.classifier.start()

        # Hand the classifier to the controller so it records each flash.
        self.stim.classifier = self.classifier

    def _init_session_logger(self):
        self.session_logger = SessionLogger(output_dir="data/sessions")

    def _init_settings_panel(self):
        self.settings_panel = SettingsPanel(
            DESIGN_WIDTH,
            DESIGN_HEIGHT
        )

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
            DESIGN_HEIGHT // target_height_cells
        )

        margin = 0

        maze_width_cells = target_width_cells
        maze_height_cells = target_height_cells

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

        self.game_manager.set_callbacks(
            on_game_over=self._on_game_over,
            on_item_collected=self._on_item_collected,
        )

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
        print("Classifier:")
        if isinstance(self.classifier, MockClassifier):
            print("  Mode: MOCK (no EEG device, random selections)")
        else:
            print("  Mode: LIVE EEG (real-time classification)")
        print()
        print("Controls:")
        print("  SPACE  - Start continuous BCI trials (auto-loops with 10s gaps)")
        print("  S      - Open settings panel")
        print("  R      - Restart game")
        print("  ESC    - Quit")
        print()
        print("=" * 50)
        print()

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

        elif key == pygame.K_SPACE:
            if self.trial_stage is TrialStage.IDLE:
                self._start_live_bci_trial()
            else:
                print("BCI trial already running")

        elif key == pygame.K_s:
            self._toggle_settings()

        elif key == pygame.K_r:
            if self.game_manager:
                self.game_manager.start_game()
                print("Game restarted")

    def _is_trial_active(self) -> bool:
        return self.trial_stage != TrialStage.IDLE

    def _start_live_bci_trial(self):

        if self._is_trial_active() and self.trial_stage not in (
            TrialStage.IDLE,
            TrialStage.BREAK,
        ):
            print("BCI trial already running")
            return

        self.trial_run_start_time = time.perf_counter()

        if self.session_logger and self.session_logger.is_active:
            self.session_logger.cancel_session()

        # One trigger session and one log file per selection.
        self.stim.triggers.stop_session()

        if self.session_logger:
            self.session_logger.start_session(
                flash_duration_ms=self.config.timing.flash_duration_ms,
                isi_ms=self.config.timing.isi_ms,
                num_sequences=self.config.timing.num_sequences,
                inter_sequence_pause_ms=0,
                flash_pattern="RANDOM",
                color_scheme=self.config.arrows.color_scheme.name,
                target_direction=None,
            )

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
        self.stim.triggers.set_current_target(None)
        self.stim.triggers.send_trial_start()

        self._start_flashing_stage()
        print("BCI trial started")

    def _start_flashing_stage(self):
        self.trial_stage = TrialStage.FLASHING
        self.trial_stage_start_time = time.perf_counter()

        # Live has no attended target. start_block also clears the classifier's
        # recorded flashes, so each selection is classified on its own block.
        self.stim.start_block(target=None)

    def _start_break_stage(self):
        self.trial_stage = TrialStage.BREAK
        self.trial_stage_start_time = time.perf_counter()
        self.stim.triggers.set_current_target(None)

    def _finalize_live_bci_trial(self, selected_direction: Direction = None):
        self.stim.triggers.send_trial_end()
        self.stim.triggers.stop_session()

        if self.session_logger and self.session_logger.is_active:
            self.session_logger.end_session(selected_direction=selected_direction)

    def _advance_trial_phase(self):
        self._start_live_bci_trial()

    def _finish_trial_run(self, cancelled: bool = False):
        was_active = self._is_trial_active()

        self.trial_stage = TrialStage.IDLE
        self.trial_stage_start_time = 0.0

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
            print("BCI trial cancelled")
        else:
            print("BCI trial complete")

    def _update_trial_run(self):
        now = time.perf_counter()
        elapsed_stage_ms = (now - self.trial_stage_start_time) * 1000.0

        if self.trial_stage == TrialStage.FLASHING:
            # Advance the flash engine. When the block finishes, the controller
            # calls _on_block_complete, which classifies and starts the break.
            self.stim.update()
            return

        if self.trial_stage == TrialStage.BREAK:
            if elapsed_stage_ms >= self.trial_break_ms:
                self._advance_trial_phase()

    def _on_block_complete(self):
        # The flash block is done. Ask the classifier for a selection (it prints
        # the result), move the player, finalize this selection, then break.
        selected_direction = self.stim.classify()

        if (
            selected_direction is not None
            and self.game_manager
            and self.game_manager.can_accept_input
        ):
            moved = self.game_manager.move_player(selected_direction)
            if not moved:
                print("  (blocked by wall)")

        self._finalize_live_bci_trial(selected_direction)
        self._start_break_stage()

    def _toggle_settings(self):
        if self.settings_panel.is_visible:
            self.settings_panel.hide()
        else:
            # Don't open settings during an active trial
            if self._is_trial_active():
                print("Cannot open settings during an active trial")
                return

            # Update panel with current values
            values = SettingsValues.from_config(self.config)
            self.settings_panel.set_values(values)
            self.settings_panel.show()
            print("Settings panel opened")

    def _on_settings_apply(self, values: SettingsValues):
        print(f"Settings applied:")
        print(f"  Flash: {values.flash_duration_ms}ms")
        print(f"  ISI: {values.isi_ms}ms")
        print(f"  Sequences: {values.num_sequences}")
        print(f"  Dullness: {values.dullness}")
        print(f"  Color: {values.color_scheme.name}")
        print(f"  SOA: {values.soa_ms}ms ({values.flash_rate_hz:.1f}Hz)")

        # Apply to config
        values.apply_to_config(self.config)

        # Rebuild the arrow surfaces for the new appearance. The classifier stays
        # attached to the controller, and timing is read live, so nothing else
        # needs rebuilding.
        self.stim.renderer.set_color_scheme(self.config.arrows.color_scheme)

        # Update game manager dullness
        if self.game_manager:
            self.game_manager.set_dullness(self.config.game.dullness)

    def _on_settings_cancel(self):
        print("Settings cancelled")

    def _on_flash_start(self, direction: Direction, sequence: int, time_ms: float):
        """Called by the controller when a flash begins - log to session.

        `time_ms` from the controller is relative to the current block. We log a
        timestamp relative to this selection's start, matching the original.
        """
        if self.session_logger and self.session_logger.is_active:
            timestamp_ms = (time.perf_counter() - self.trial_run_start_time) * 1000.0
            self.session_logger.log_flash_start(direction, sequence, timestamp_ms)

    def _on_flash_end(self, direction: Direction, sequence: int, time_ms: float):
        """Called by the controller when a flash ends - log to session."""
        if self.session_logger and self.session_logger.is_active:
            timestamp_ms = (time.perf_counter() - self.trial_run_start_time) * 1000.0
            self.session_logger.log_flash_end(direction, sequence, timestamp_ms)

    def _is_game_over(self) -> bool:
        return (
            self.game_manager is not None
            and self.game_manager._state == GameState.GAME_OVER
        )

    def _on_game_over(self, stats):
        print(f"Game finished! Score: {stats.score}")
        # Stop any active BCI trial so arrows freeze.
        self._finish_trial_run(cancelled=False)

    def _on_item_collected(self, points: int):
        pass  # Take care of this later

    def _update(self):
        delta_ms = self.clock.get_time()

        if not self._is_game_over():
            if self._is_trial_active():
                self._update_trial_run()

        if self.game_manager:
            self.game_manager.update(delta_ms)

    def _draw(self):
        self.render_surface.fill(self.config.display.background_color)

        if self.game_manager:
            self.game_manager.draw(self.render_surface)

        if not self._is_game_over():
            # Arrows show while idle and flashing; during the break they are
            # hidden and only the "Get ready..." overlay appears.
            if self.trial_stage in (TrialStage.IDLE, TrialStage.FLASHING):
                self.stim.draw(self.render_surface)
            self._draw_trial_overlay()

        if self.settings_panel:
            self.settings_panel.draw(self.render_surface)

        self.screen.fill((0, 0, 0))
        if self.scale_factor == 1.0 and self.render_offset == (0, 0):
            self.screen.blit(self.render_surface, (0, 0))
        else:
            scaled = pygame.transform.smoothscale(self.render_surface, self.scaled_size)
            self.screen.blit(scaled, self.render_offset)

        pygame.display.flip()

    def _draw_trial_overlay(self):
        if self.trial_stage in (TrialStage.IDLE, TrialStage.FLASHING):
            return

        panel_rect = self.stim.get_panel_rect()
        center = panel_rect.center if panel_rect else (DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2)

        if self.trial_stage == TrialStage.BREAK:
            elapsed_ms = (time.perf_counter() - self.trial_stage_start_time) * 1000.0
            remaining_ms = self.trial_break_ms - elapsed_ms
            if remaining_ms <= 3000:
                text = self.font_large.render("Get ready...", True, (130, 130, 130))
                text_rect = text.get_rect(center=center)
                self.render_surface.blit(text, text_rect)

    def _cleanup(self):
        if self._is_trial_active():
            self._finish_trial_run(cancelled=True)
        elif self.session_logger and self.session_logger.is_active:
            self.session_logger.cancel_session()

        if self.classifier:
            self.classifier.stop()

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
        default=1920,
        help="Window width (default: 1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1200,
        help="Window height (default: 1200)"
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

    # The original live mode flashed with no pause between sequences (its
    # hand-rolled loop ignored inter_sequence_pause_ms). The timing engine
    # honors this value, so set it to 0 to keep the flash timing identical.
    # Remove this line if you want the configured inter-sequence pause instead.
    config.timing.inter_sequence_pause_ms = 0

    if args.sequences:
        config.timing.num_sequences = args.sequences

    try:
        app = Application(config)

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