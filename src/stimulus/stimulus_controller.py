import time
from typing import Optional, Callable, Dict

import pygame

from config import Config, Direction
from src.stimulus.arrow_renderer import ArrowRenderer
from src.stimulus.timing_controller import TimingController
from src.stimulus.triggers import TriggerManager


class StimulusController:

    def __init__(self, config: Config):
        self.config = config

        self.renderer = ArrowRenderer(config.arrows, config.layout)
        self.timing = TimingController(config.timing)
        self.triggers = TriggerManager(config.triggers)

        self.classifier = None

        self._flash_states: Dict[Direction, bool] = {d: False for d in Direction.all()}

        self._on_flash_start: Optional[Callable[[Direction, int, float], None]] = None
        self._on_flash_end: Optional[Callable[[Direction, int, float], None]] = None
        self._on_block_complete: Optional[Callable[[], None]] = None

        self.timing.set_callbacks(
            on_flash_start=self._handle_flash_start,
            on_flash_end=self._handle_flash_end,
            on_selection_complete=self._handle_block_complete,
        )

    def initialize(self, screen_width: int, screen_height: int):
        self.renderer.initialize(screen_width, screen_height)

    def shutdown(self):
        self.triggers.stop_session()

    def set_callbacks(
        self,
        on_flash_start: Optional[Callable[[Direction, int, float], None]] = None,
        on_flash_end: Optional[Callable[[Direction, int, float], None]] = None,
        on_block_complete: Optional[Callable[[], None]] = None,
    ):
        self._on_flash_start = on_flash_start
        self._on_flash_end = on_flash_end
        self._on_block_complete = on_block_complete

    def start_block(self, target: Optional[Direction] = None):
        self._flash_states = {d: False for d in Direction.all()}
        self.triggers.set_current_target(target)

        if self.classifier is not None:
            self.classifier.clear_events()

        self.timing.start()

    def update(self):
        self.timing.update()

    def draw(self, surface: pygame.Surface):
        self.renderer.draw(surface, self._flash_states)

    def classify(self) -> Optional[Direction]:
        if self.classifier is None:
            return None

        result = self.classifier.classify_trial()
        if not result:
            print("BCI classification failed for this block.")
            return None

        selected = Direction(result["direction"])
        print(
            f"BCI Selection: {selected.value.upper()} "
            f"(confidence={result['confidence']:.3f}, "
            f"epochs={result['n_epochs_used']}/{result['n_epochs_total']})"
        )
        return selected

    def _handle_flash_start(self, direction: Direction, sequence: int, time_ms: float):
        self._flash_states[direction] = True
        self.triggers.send_flash(direction)

        if self.classifier is not None:
            try:
                from pylsl import local_clock
                timestamp = local_clock()
            except ImportError:
                timestamp = time.perf_counter()
            self.classifier.record_flash(direction=direction.value, timestamp=timestamp)

        if self._on_flash_start:
            self._on_flash_start(direction, sequence, time_ms)

    def _handle_flash_end(self, direction: Direction, sequence: int, time_ms: float):
        self._flash_states[direction] = False
        if self._on_flash_end:
            self._on_flash_end(direction, sequence, time_ms)

    def _handle_block_complete(self):
        if self._on_block_complete:
            self._on_block_complete()

    @property
    def is_flashing(self) -> bool:
        return self.timing.is_running

    @property
    def stats(self):
        return self.timing.stats

    def get_panel_rect(self) -> Optional[pygame.Rect]:
        return self.renderer.get_panel_rect()

def demo():
    from config import Config

    config = Config()
    config.timing.num_sequences = 2  # keep the demo short

    stim = StimulusController(config)

    def on_flash_start(direction, seq, time_ms):
        print(f"  {time_ms:8.1f}ms: {direction.value:5} ON  (seq {seq})")

    def on_flash_end(direction, seq, time_ms):
        print(f"  {time_ms:8.1f}ms: {direction.value:5} OFF")

    def on_block_complete():
        print("  === Block complete ===")

    stim.set_callbacks(
        on_flash_start=on_flash_start,
        on_flash_end=on_flash_end,
        on_block_complete=on_block_complete,
    )

    print("StimulusController demo (no window, no EEG)")
    stim.triggers.start_session(
        flash_duration_ms=config.timing.flash_duration_ms,
        isi_ms=config.timing.isi_ms,
        num_sequences=config.timing.num_sequences,
    )
    stim.start_block(target=Direction.UP)

    while stim.is_flashing:
        stim.update()
        time.sleep(0.001)

    stim.triggers.stop_session()

    s = stim.stats
    print(f"\n  Max timing error: {s.max_error_ms:.3f} ms  (acceptable: {s.is_acceptable})")


if __name__ == "__main__":
    demo()