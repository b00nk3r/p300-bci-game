"""
Arrow Renderer
==============
Handles drawing of arrow stimuli with configurable appearance.

Design based on P300 BCI research recommendations:
- Arrow size: ~1° visual angle (~100px at 60cm viewing distance)
- High contrast between idle and flash states
- Semi-transparent panel behind arrows for visual grouping
"""

import pygame
import math
from typing import Dict, Tuple, Optional

from config import ArrowConfig, LayoutConfig, Direction, ColorScheme


class ArrowRenderer:
    """
    Renders arrow stimuli for P300 BCI interface.
    
    Features:
    - Pre-rendered arrow surfaces for performance
    - Multiple color schemes (Gray/White, Green/Blue, Inverted)
    - Semi-transparent background panel
    - Smooth, filled arrow shapes
    
    Usage:
        renderer = ArrowRenderer(arrow_config, layout_config)
        renderer.initialize(screen_width, screen_height)
        
        # In game loop:
        flash_states = {Direction.UP: True, Direction.DOWN: False, ...}
        renderer.draw(screen, flash_states)
    """
    
    def __init__(self, arrow_config: ArrowConfig, layout_config: LayoutConfig):
        self.arrow_config = arrow_config
        self.layout_config = layout_config
        
        # Screen dimensions (set on initialize)
        self._screen_width: int = 0
        self._screen_height: int = 0
        
        # Arrow positions (calculated on initialize)
        self._positions: Dict[Direction, Tuple[int, int]] = {}
        
        # Pre-rendered surfaces
        self._idle_surfaces: Dict[Direction, pygame.Surface] = {}
        self._flash_surfaces: Dict[Direction, pygame.Surface] = {}
        self._panel_surface: Optional[pygame.Surface] = None
        self._panel_rect: Optional[pygame.Rect] = None
        
        # Color cache
        self._idle_color: Tuple[int, int, int] = (128, 128, 128)
        self._flash_color: Tuple[int, int, int] = (255, 255, 255)
        self._panel_color: Tuple[int, int, int] = (0, 0, 0)
        
    def initialize(self, screen_width: int, screen_height: int):
        """
        Initialize renderer with screen dimensions.
        Must be called after pygame.display.set_mode().
        
        Args:
            screen_width: Width of the display in pixels
            screen_height: Height of the display in pixels
        """
        self._screen_width = screen_width
        self._screen_height = screen_height
        
        # Calculate arrow positions
        self._positions = self.layout_config.get_positions(screen_width, screen_height)
        
        # Set colors based on scheme
        self._update_colors()
        
        # Create all surfaces
        self._create_panel_surface()
        self._create_arrow_surfaces()
        
    def _update_colors(self):
        """Update colors based on current color scheme"""
        scheme = self.arrow_config.color_scheme
        
        if scheme == ColorScheme.GRAY_WHITE:
            self._idle_color = (128, 128, 128)   # Gray
            self._flash_color = (255, 255, 255)  # White
            self._panel_color = (0, 0, 0)        # Black
            
        elif scheme == ColorScheme.GREEN_BLUE:
            self._idle_color = (0, 64, 128)      # Dark blue
            self._flash_color = (0, 255, 128)    # Bright green
            self._panel_color = (0, 0, 0)        # Black
            
        elif scheme == ColorScheme.INVERTED:
            self._idle_color = (200, 200, 200)   # Light gray
            self._flash_color = (40, 40, 40)     # Dark
            self._panel_color = (240, 240, 240)  # Light
            
    def _create_panel_surface(self):
        """Create the semi-transparent panel behind arrows"""
        # Calculate panel bounds based on arrow positions
        positions = list(self._positions.values())
        
        if not positions:
            return
            
        # Find bounding box of all arrows
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        padding = self.arrow_config.panel_padding + self.arrow_config.size // 2
        
        left = min(xs) - padding
        right = max(xs) + padding
        top = min(ys) - padding
        bottom = max(ys) + padding
        
        width = right - left
        height = bottom - top
        
        # Create panel surface with alpha
        self._panel_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Fill with semi-transparent color
        panel_color_with_alpha = (*self._panel_color, self.arrow_config.panel_alpha)
        self._panel_surface.fill(panel_color_with_alpha)
        
        # Optional: Add subtle border
        border_color = tuple(min(255, c + 30) for c in self._panel_color) + (self.arrow_config.panel_alpha,)
        pygame.draw.rect(self._panel_surface, border_color, 
                         self._panel_surface.get_rect(), 2)
        
        # Store panel position
        self._panel_rect = pygame.Rect(left, top, width, height)
        
    def _create_arrow_surfaces(self):
        """Pre-render arrow surfaces for each direction and state"""
        size = self.arrow_config.size
        
        for direction in Direction.all():
            # Create idle surface
            self._idle_surfaces[direction] = self._create_arrow_surface(
                direction, self._idle_color, size
            )
            
            # Create flash surface
            self._flash_surfaces[direction] = self._create_arrow_surface(
                direction, self._flash_color, size
            )
            
    def _create_arrow_surface(
        self, 
        direction: Direction, 
        color: Tuple[int, int, int],
        size: int
    ) -> pygame.Surface:
        """
        Create a single arrow surface.
        
        Args:
            direction: Which way the arrow points
            color: RGB color tuple
            size: Size of the arrow in pixels
            
        Returns:
            pygame.Surface with the arrow drawn on it
        """
        # Create surface with transparency
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Calculate arrow points
        points = self._get_arrow_points(direction, size)
        
        # Draw filled arrow
        pygame.draw.polygon(surface, color, points)
        
        # Optional: Add subtle outline for depth
        outline_color = tuple(max(0, c - 30) for c in color)
        pygame.draw.polygon(surface, outline_color, points, 2)
        
        return surface
        
    def _get_arrow_points(
        self, 
        direction: Direction, 
        size: int
    ) -> list:
        """
        Calculate polygon points for an arrow shape.
        
        The arrow is drawn as a triangle pointing in the specified direction.
        
        Args:
            direction: Which way the arrow points
            size: Size of the bounding box
            
        Returns:
            List of (x, y) tuples for polygon vertices
        """
        # Margins from edge
        margin = size * 0.1
        
        # Arrow dimensions
        tip = margin                    # Tip position from edge
        base = size - margin            # Base position from edge
        center = size / 2               # Center line
        width = size * 0.35             # Half-width of arrow base
        
        if direction == Direction.UP:
            return [
                (center, tip),              # Tip (top)
                (center - width, base),     # Bottom left
                (center + width, base),     # Bottom right
            ]
            
        elif direction == Direction.DOWN:
            return [
                (center, base),             # Tip (bottom)
                (center - width, tip),      # Top left
                (center + width, tip),      # Top right
            ]
            
        elif direction == Direction.LEFT:
            return [
                (tip, center),              # Tip (left)
                (base, center - width),     # Top right
                (base, center + width),     # Bottom right
            ]
            
        elif direction == Direction.RIGHT:
            return [
                (base, center),             # Tip (right)
                (tip, center - width),      # Top left
                (tip, center + width),      # Bottom left
            ]
            
        return []
        
    def draw(self, screen: pygame.Surface, flash_states: Dict[Direction, bool]):
        """
        Draw all arrows with current flash states.
        
        Args:
            screen: Pygame surface to draw on
            flash_states: Dict mapping Direction to bool (True = flashing)
        """
        # Draw panel background first
        self._draw_panel(screen)
        
        # Draw each arrow
        for direction in Direction.all():
            is_flashing = flash_states.get(direction, False)
            self._draw_arrow(screen, direction, is_flashing)
            
    def _draw_panel(self, screen: pygame.Surface):
        """Draw the background panel"""
        if self._panel_surface is not None and self._panel_rect is not None:
            screen.blit(self._panel_surface, self._panel_rect.topleft)
            
    def _draw_arrow(
        self, 
        screen: pygame.Surface, 
        direction: Direction, 
        is_flashing: bool
    ):
        """
        Draw a single arrow.
        
        Args:
            screen: Surface to draw on
            direction: Which arrow to draw
            is_flashing: Whether the arrow is currently flashing
        """
        # Get appropriate surface
        if is_flashing:
            surface = self._flash_surfaces.get(direction)
        else:
            surface = self._idle_surfaces.get(direction)
            
        if surface is None:
            return
            
        # Get position (center of arrow)
        position = self._positions.get(direction)
        if position is None:
            return
            
        # Calculate top-left corner for blitting
        rect = surface.get_rect(center=position)
        
        # Draw
        screen.blit(surface, rect)
        
    def set_color_scheme(self, scheme: ColorScheme):
        """
        Change the color scheme at runtime.
        
        Args:
            scheme: New color scheme to use
        """
        self.arrow_config.color_scheme = scheme
        self._update_colors()
        self._create_panel_surface()
        self._create_arrow_surfaces()
        
    def get_panel_rect(self) -> Optional[pygame.Rect]:
        """
        Get the rectangle of the arrow panel.
        Useful for positioning game elements around it.
        
        Returns:
            pygame.Rect of the panel, or None if not initialized
        """
        return self._panel_rect
        
    def get_arrow_positions(self) -> Dict[Direction, Tuple[int, int]]:
        """
        Get the center positions of all arrows.
        
        Returns:
            Dict mapping Direction to (x, y) center coordinates
        """
        return self._positions.copy()


# =============================================================================
# Testing / Demo
# =============================================================================

def demo():
    """Run a visual demo of the arrow renderer"""
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 3)[0])  # Add project root
    
    from config import Config
    
    pygame.init()
    
    config = Config()
    config.display.width = 1024
    config.display.height = 768
    
    screen = pygame.display.set_mode(
        (config.display.width, config.display.height)
    )
    pygame.display.set_caption("Arrow Renderer Demo")
    
    # Create renderer
    renderer = ArrowRenderer(config.arrows, config.layout)
    renderer.initialize(config.display.width, config.display.height)
    
    clock = pygame.time.Clock()
    running = True
    
    # Flash states (toggle with arrow keys)
    flash_states = {d: False for d in Direction.all()}
    
    # For automatic demo
    auto_mode = True
    auto_timer = 0
    auto_index = 0
    directions = Direction.all()
    
    font = pygame.font.Font(None, 32)
    
    print("Arrow Renderer Demo")
    print("  Arrow keys: Toggle individual arrows")
    print("  1/2/3: Switch color scheme")
    print("  A: Toggle auto mode")
    print("  ESC: Quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
                # Toggle individual arrows
                elif event.key == pygame.K_UP:
                    flash_states[Direction.UP] = not flash_states[Direction.UP]
                elif event.key == pygame.K_DOWN:
                    flash_states[Direction.DOWN] = not flash_states[Direction.DOWN]
                elif event.key == pygame.K_LEFT:
                    flash_states[Direction.LEFT] = not flash_states[Direction.LEFT]
                elif event.key == pygame.K_RIGHT:
                    flash_states[Direction.RIGHT] = not flash_states[Direction.RIGHT]
                    
                # Color schemes
                elif event.key == pygame.K_1:
                    renderer.set_color_scheme(ColorScheme.GRAY_WHITE)
                    print("Switched to GRAY_WHITE scheme")
                elif event.key == pygame.K_2:
                    renderer.set_color_scheme(ColorScheme.GREEN_BLUE)
                    print("Switched to GREEN_BLUE scheme")
                elif event.key == pygame.K_3:
                    renderer.set_color_scheme(ColorScheme.INVERTED)
                    print("Switched to INVERTED scheme")
                    
                # Toggle auto mode
                elif event.key == pygame.K_a:
                    auto_mode = not auto_mode
                    if not auto_mode:
                        flash_states = {d: False for d in Direction.all()}
                    print(f"Auto mode: {auto_mode}")
                    
        # Auto flash demo
        if auto_mode:
            auto_timer += clock.get_time()
            if auto_timer >= 225:  # SOA of 225ms
                # Turn off previous
                flash_states = {d: False for d in Direction.all()}
                # Flash next arrow for 100ms
                flash_states[directions[auto_index]] = True
                auto_index = (auto_index + 1) % len(directions)
                auto_timer = 0
                
            # Turn off after 100ms flash duration
            if auto_timer >= 100:
                flash_states = {d: False for d in Direction.all()}
        
        # Draw
        screen.fill(config.display.background_color)
        renderer.draw(screen, flash_states)
        
        # Draw instructions
        instructions = [
            "Arrow Renderer Demo",
            "",
            "Arrow keys: Toggle flash",
            "1/2/3: Color scheme",
            "A: Auto mode" + (" (ON)" if auto_mode else " (OFF)"),
            "ESC: Quit",
        ]
        
        y = 10
        for line in instructions:
            text = font.render(line, True, (80, 80, 80))
            screen.blit(text, (10, y))
            y += 30
            
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()


if __name__ == "__main__":
    demo()