"""
Arrow Renderer
==============
Handles drawing of arrow stimuli with configurable appearance.
"""

import pygame
from typing import Tuple, Optional

from config import ArrowConfig, LayoutConfig, Direction, ColorScheme


class ArrowRenderer:
    """
    Renders arrow stimuli for P300 BCI interface.
    
    Responsibilities:
    - Draw arrows in idle and flash states
    - Handle different color schemes
    - Draw background panel
    """
    
    def __init__(self, arrow_config: ArrowConfig, layout_config: LayoutConfig):
        self.arrow_config = arrow_config
        self.layout_config = layout_config
        
        # Surfaces (created on initialize)
        self._arrow_surfaces: dict = {}
        self._panel_surface: Optional[pygame.Surface] = None
        
        # Screen dimensions
        self._screen_width = 0
        self._screen_height = 0
        self._positions: dict = {}
        
    def initialize(self, screen_width: int, screen_height: int):
        """Initialize renderer with screen dimensions"""
        self._screen_width = screen_width
        self._screen_height = screen_height
        
        # Calculate arrow positions
        self._positions = self.layout_config.get_positions(screen_width, screen_height)
        
        # Create arrow surfaces
        self._create_arrow_surfaces()
        
        # Create panel surface
        self._create_panel_surface()
        
    def _create_arrow_surfaces(self):
        """Pre-render arrow surfaces for each direction and state"""
        # TODO: Create arrow surfaces for idle and flash states
        # For each direction: UP, DOWN, LEFT, RIGHT
        # For each state: idle, flash
        pass
        
    def _create_panel_surface(self):
        """Create the semi-transparent panel behind arrows"""
        # TODO: Create panel surface
        pass
        
    def draw(self, screen: pygame.Surface, flash_states: dict):
        """
        Draw all arrows with current flash states.
        
        Args:
            screen: Pygame surface to draw on
            flash_states: Dict mapping Direction to bool (True = flashing)
        """
        # TODO: Draw panel
        # TODO: Draw each arrow with appropriate state
        pass
        
    def draw_arrow(
        self, 
        screen: pygame.Surface, 
        direction: Direction, 
        is_flashing: bool
    ):
        """Draw a single arrow"""
        # TODO: Draw arrow at correct position with correct color
        pass
