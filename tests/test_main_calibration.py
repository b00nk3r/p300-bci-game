"""Regression tests for calibration phase control."""

from types import SimpleNamespace

import pytest

pygame = pytest.importorskip("pygame")

from config import Config, Direction
from main import Application, CalibrationStage


class TriggerRecorder:
    """Capture attended targets sent to the trigger logger."""

    def __init__(self):
        self.targets = []

    def set_current_target(self, target):
        self.targets.append(target)


def make_app():
    """Create an application instance without initializing pygame."""
    app = Application(Config())
    app.arrow_manager = SimpleNamespace(
        triggers=TriggerRecorder(),
        is_active=False,
    )
    return app


def test_space_from_waiting_advances_phase(monkeypatch):
    """SPACE in WAITING should advance instead of replaying the same phase."""
    app = make_app()
    app.calibration_stage = CalibrationStage.WAITING

    calls = []
    monkeypatch.setattr(app, "_advance_calibration_phase", lambda: calls.append("advance"))
    monkeypatch.setattr(app, "_start_calibration_run", lambda: calls.append("start"))

    app._handle_keydown(pygame.K_SPACE)

    assert calls == ["advance"]


def test_start_flashing_stage_restores_current_target():
    """Flashing should restore the attended target before flash logging begins."""
    app = make_app()
    app.calibration_phase_order = Direction.all()
    app.calibration_phase_index = 2

    app._start_flashing_stage()

    assert app.calibration_stage == CalibrationStage.FLASHING
    assert app.arrow_manager.triggers.targets[-1] == Direction.LEFT
    assert len(app.calibration_flash_plan) == app.config.timing.num_sequences * len(Direction.all())


def test_arrow_from_idle_starts_live_bci_trial(monkeypatch):
    """Live BCI mode should use the pressed arrow as the trial target."""
    app = make_app()
    app.classifier = object()

    started = []
    manual_moves = []
    monkeypatch.setattr(app, "_start_live_bci_trial", lambda direction: started.append(direction))
    monkeypatch.setattr(app, "_manual_move", lambda direction: manual_moves.append(direction))

    app._handle_keydown(pygame.K_RIGHT)

    assert started == [Direction.RIGHT]
    assert manual_moves == []


def test_arrow_from_waiting_starts_next_live_bci_trial(monkeypatch):
    """WAITING in live BCI mode should resume with the next target arrow."""
    app = make_app()
    app.classifier = object()
    app.calibration_stage = CalibrationStage.WAITING

    started = []
    manual_moves = []
    monkeypatch.setattr(app, "_start_live_bci_trial", lambda direction: started.append(direction))
    monkeypatch.setattr(app, "_manual_move", lambda direction: manual_moves.append(direction))

    app._handle_keydown(pygame.K_LEFT)

    assert started == [Direction.LEFT]
    assert manual_moves == []
