import time
import random
from enum import Enum, auto

import pygame

from config import Direction
from src.app.application import Application, DESIGN_WIDTH, DESIGN_HEIGHT
from src.game.game_manager import GameManagerConfig, GameState


class TrialStage(Enum):
    IDLE = auto()
    FLASHING = auto()
    BREAK = auto()


class MockClassifier:
    """Stand-in classifier used when no EEG device is available.

    Records the flashes of a block, then returns a random one of the flashed
    directions so the game loop can be exercised without a headset.
    """

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
        directions = list(set(e["direction"] for e in self._events))
        chosen = random.choice(directions)
        return {
            "direction": chosen,
            "confidence": round(random.uniform(0.4, 0.95), 3),
            "n_epochs_used": len(self._events),
            "n_epochs_total": len(self._events),
        }


class LiveMode(Application):
    """Live BCI mode.

    Continuous selections: SPACE starts a trial, the stimulus controller flashes
    a block, the classifier picks a direction, the player moves, then a 10-second
    break and the next trial begins. Each selection opens and closes its own
    trigger session and log file.
    """

    def __init__(self, config):
        super().__init__(config)
        self.classifier = None
        self.trial_stage = TrialStage.IDLE
        self.trial_stage_start_time = 0.0
        self.trial_break_ms = 10000  # 10-second gap between selections

    # ===================================================================
    # Mode hooks
    # ===================================================================

    def _init_extra(self):
        self._init_classifier()

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

    def _build_game_config(self, maze_width_cells, maze_height_cells, cell_size, use_corridors):
        return GameManagerConfig(
            base_maze_width=maze_width_cells,
            base_maze_height=maze_height_cells,
            max_maze_width=maze_width_cells,
            max_maze_height=maze_height_cells,
            base_collectibles=5,
            max_collectibles=5,
            cell_size=cell_size,
            use_corridors=use_corridors,
        )

    def _register_game_callbacks(self):
        self.game_manager.set_callbacks(
            on_game_over=self._on_game_over,
            on_item_collected=self._on_item_collected,
        )

    def _print_mode_info(self):
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

    def _handle_mode_key(self, key):
        if key == pygame.K_SPACE:
            if self.trial_stage is TrialStage.IDLE:
                self._start_live_bci_trial()
            else:
                print("BCI trial already running")
        elif key == pygame.K_r:
            if self.game_manager:
                self.game_manager.start_game()
                print("Game restarted")

    def _is_active(self) -> bool:
        return self.trial_stage != TrialStage.IDLE

    def _update_mode(self):
        if not self._is_game_over():
            if self._is_active():
                self._update_trial_run()

    def _draw_mode(self, surface):
        if not self._is_game_over():
            # Arrows show while idle and flashing; during the break they are
            # hidden and only the "Get ready..." overlay appears.
            if self.trial_stage in (TrialStage.IDLE, TrialStage.FLASHING):
                self.stim.draw(surface)
            self._draw_trial_overlay(surface)

    def _on_block_complete(self):
        # The flash block is done. Ask the classifier (it prints the result),
        # move the player, finalize this selection, then break.
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

    def _on_cleanup(self):
        if self._is_active():
            self._finish_trial_run(cancelled=True)
        elif self.session_logger and self.session_logger.is_active:
            self.session_logger.cancel_session()
        if self.classifier:
            self.classifier.stop()

    # ===================================================================
    # Trial state machine
    # ===================================================================

    def _start_live_bci_trial(self):
        if self._is_active() and self.trial_stage not in (
            TrialStage.IDLE,
            TrialStage.BREAK,
        ):
            print("BCI trial already running")
            return

        self.session_log_origin = time.perf_counter()

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
        was_active = self._is_active()

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

    def _draw_trial_overlay(self, surface):
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
                surface.blit(text, text_rect)

    # ===================================================================
    # Game callbacks
    # ===================================================================

    def _is_game_over(self) -> bool:
        return (
            self.game_manager is not None
            and self.game_manager.state == GameState.GAME_OVER
        )

    def _on_game_over(self, stats):
        print(f"Game finished! Score: {stats.score}")
        # Stop any active BCI trial so arrows freeze.
        self._finish_trial_run(cancelled=False)