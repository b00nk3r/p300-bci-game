"""
Arrow Manager
=============
Coordinates arrow rendering and timing for P300 BCI interface.
"""

import pygame
from typing import Optional, Callable
from enum import Enum, auto

from config import Config, Direction
from src.stimulus.arrow_renderer import ArrowRenderer
from src.stimulus.timing_controller import TimingController


class SelectionState(Enum):
    """State of the BCI selection process"""
    IDLE = auto()
    FLASHING = auto()
    PROCESSING = auto()
    FEEDBACK = auto()


class ArrowManager:
    """
    Main interface for the P300 arrow stimulus system.
    
    Coordinates:
    - Arrow rendering
    - Flash timing
    - Trigger output
    - Selection state
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Sub-components
        self.renderer = ArrowRenderer(config.arrows, config.layout)
        self.timing = TimingController(config.timing)
        
        # State
        self._state = SelectionState.IDLE
        self._flash_states = {d: False for d in Direction.all()}
        self._selected_direction: Optional[Direction] = None
        
        # Callbacks
        self._on_selection_complete: Optional[Callable[[Direction], None]] = None
        
        # Setup timing callbacks
        self.timing.set_callbacks(
            on_flash_start=self._handle_flash_start,
            on_flash_end=self._handle_flash_end,
            on_selection_complete=self._handle_timing_complete,
        )
        
    def initialize(self, screen_width: int, screen_height: int):
        """Initialize with screen dimensions"""
        self.renderer.initialize(screen_width, screen_height)
        
    def start_selection(self):
        """Begin a new BCI selection (start flashing)"""
        if self._state != SelectionState.IDLE:
            return
            
        self._state = SelectionState.FLASHING
        self._flash_states = {d: False for d in Direction.all()}
        self.timing.start_selection()
        
        # TODO: Send trial start trigger
        
    def stop_selection(self):
        """Stop the current selection"""
        self._state = SelectionState.IDLE
        self.timing.stop_selection()
        self._flash_states = {d: False for d in Direction.all()}
        
    def update(self):
        """Update state (call every frame)"""
        if self._state == SelectionState.FLASHING:
            self.timing.update()
            
    def draw(self, screen: pygame.Surface):
        """Draw arrows (call every frame)"""
        self.renderer.draw(screen, self._flash_states)
        
    def _handle_flash_start(self, direction: Direction):
        """Called when a flash starts"""
        self._flash_states[direction] = True
        # TODO: Send flash trigger
        
    def _handle_flash_end(self, direction: Direction):
        """Called when a flash ends"""
        self._flash_states[direction] = False
        
    def _handle_timing_complete(self):
        """Called when all sequences are done"""
        self._state = SelectionState.PROCESSING
        # TODO: Wait for classifier result
        
    def set_selection_callback(self, callback: Callable[[Direction], None]):
        """Set callback for when selection is complete"""
        self._on_selection_complete = callback
        
    @property
    def is_active(self) -> bool:
        """Whether selection is in progress"""
        return self._state == SelectionState.FLASHING
