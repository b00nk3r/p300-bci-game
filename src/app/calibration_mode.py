import time
import random
from enum import Enum, auto

import pygame

from config import Direction
from src.app.application import Application, DESIGN_WIDTH, DESIGN_HEIGHT
from src.game.game_manager import GameManagerConfig


class CalibrationStage(Enum):
    IDLE = auto()
    INSTRUCTION = auto()
    FLASHING = auto()
    BREAK = auto()


class CalibrationMode(Application):
    """Calibration data-collection mode.

    Walks through 12 phases (3 shuffled blocks of the 4 directions). Each phase
    shows an ATTEND instruction, flashes one block through the stimulus
    controller, then a break, then advances. No classifier; one trigger session
    and one log file for the whole run.
    """

    def __init__(self, config):
        super().__init__(config)
        self.calibration_stage = CalibrationStage.IDLE
        self.calibration_phase_order = []
        self.calibration_phase_index = 0
        self.calibration_stage_start_time = 0.0
        self.calibration_instruction_ms = 2000
        self.calibration_break_ms = 2000

    # ===================================================================
    # Mode hooks
    # ===================================================================

    def _build_game_config(self, maze_width_cells, maze_height_cells, cell_size, use_corridors):
        return GameManagerConfig(
            base_maze_width=maze_width_cells,
            base_maze_height=maze_height_cells,
            max_maze_width=maze_width_cells,   # Don't grow beyond screen
            max_maze_height=maze_height_cells,
            maze_growth_per_level=0,           # Keep same size, just regenerate
            base_collectibles=2,               # Fixed donut count per level
            collectibles_per_level=0,
            max_collectibles=2,
            cell_size=cell_size,
            use_corridors=use_corridors,
        )

    def _register_game_callbacks(self):
        self.game_manager.set_callbacks(
            on_level_complete=self._on_level_complete,
            on_item_collected=self._on_item_collected,
        )

    def _print_mode_info(self):
        print("Controls:")
        print("  S      - Open settings panel")
        print("  R      - Restart current level")
        print("  N      - Skip to next level")
        print("  ESC    - Quit")
        print("  SPACE  - Start calibration run")
        print()

    def _handle_mode_key(self, key):
        if key == pygame.K_SPACE:
            if self.calibration_stage == CalibrationStage.IDLE:
                self._start_calibration_run()
            elif self._is_active():
                print("Calibration already running")
        elif key == pygame.K_r:
            if self._is_active():
                print("Cannot restart during an active trial")
                return
            elif self.game_manager:
                self.game_manager.restart_level()
                print("Level restarted")
        elif key == pygame.K_n:
            if self._is_active():
                print("Cannot skip levels during an active trial")
                return
            if self.game_manager:
                self.game_manager.next_level()
                print(f"Advanced to level {self.game_manager.stats.level}")

    def _is_active(self) -> bool:
        return self.calibration_stage != CalibrationStage.IDLE

    def _update_mode(self):
        if self._is_active():
            self._update_calibration_run()

    def _draw_mode(self, surface):
        # Panel plus arrows in their current flash states (idle when not flashing).
        self.stim.draw(surface)
        # ATTEND overlay during the instruction stage.
        if (
            self.calibration_stage == CalibrationStage.INSTRUCTION
            and self.calibration_phase_index < len(self.calibration_phase_order)
        ):
            self._draw_attend_overlay(surface)

    def _on_block_complete(self):
        self._start_break_stage()

    def _on_cleanup(self):
        if self._is_active():
            self._finish_calibration_run(cancelled=True)
        elif self.session_logger and self.session_logger.is_active:
            self.session_logger.cancel_session()

    # ===================================================================
    # Calibration state machine
    # ===================================================================

    def _start_calibration_run(self):
        if self._is_active():
            print("Calibration already running")
            return

        self.calibration_phase_order = []
        for _ in range(3):
            block = Direction.all()
            random.shuffle(block)
            self.calibration_phase_order.extend(block)

        self.calibration_phase_index = 0
        self.session_log_origin = time.perf_counter()

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
        attended = self.calibration_phase_order[self.calibration_phase_index]

        self.calibration_stage = CalibrationStage.FLASHING
        self.calibration_stage_start_time = time.perf_counter()

        # The controller flashes num_sequences sequences of the four arrows and
        # reports each flash and the block completion through the base callbacks.
        self.stim.start_block(target=attended)

    def _start_break_stage(self):
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
        was_active = self._is_active()

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
            self.stim.update()
            return

        if self.calibration_stage == CalibrationStage.BREAK:
            if elapsed_stage_ms >= self.calibration_break_ms:
                self._advance_calibration_phase()

    def _draw_attend_overlay(self, surface):
        panel_rect = self.stim.get_panel_rect()
        center = panel_rect.center if panel_rect else (DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2)
        attended = self.calibration_phase_order[self.calibration_phase_index]
        message = f"ATTEND {attended.value.upper()}"
        text = self.font_large.render(message, True, (200, 200, 200))
        text_rect = text.get_rect(center=center)
        surface.blit(text, text_rect)

    # ===================================================================
    # Game callbacks
    # ===================================================================

    def _on_level_complete(self, level, score):
        print(f"Level {level} complete! Score: {score}")
