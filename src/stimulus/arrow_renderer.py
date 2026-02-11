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
        """Update colors based on current color scheme (all grayscale)"""
        scheme = self.arrow_config.color_scheme
        
        if scheme == ColorScheme.GRAY_WHITE:
            self._idle_color = (128, 128, 128)   # Medium gray
            self._flash_color = (255, 255, 255)  # Brightest white
            self._panel_color = (0, 0, 0)        # Black
            
        elif scheme == ColorScheme.GREEN_BLUE:
            # Now grayscale variant: darker idle, brightest white flash
            self._idle_color = (80, 80, 80)      # Dark gray
            self._flash_color = (255, 255, 255)  # Brightest white
            self._panel_color = (0, 0, 0)        # Black
            
        elif scheme == ColorScheme.INVERTED:
            # Inverted grayscale: light idle, dark flash
            self._idle_color = (200, 200, 200)   # Light gray
            self._flash_color = (255, 255, 255)  # Brightest white (changed from dark)
            self._panel_color = (40, 40, 40)     # Dark gray panel
            
    def _create_panel_surface(self):
        """
        Create a full rectangular black panel behind arrows.

        This keeps arrow positions/sizes unchanged while making the entire
        center square area uniformly black.
        """
        positions = self._positions
        
        if not positions:
            return
        
        panel_size = self.arrow_config.panel_size  # 200px
        half_panel = panel_size // 2
        
        # Calculate panel bounds around the arrow cluster.
        # Slightly expand to reduce the visible border gap against the game cutout.
        panel_expand_px = 20
        left = positions[Direction.LEFT][0] - half_panel
        right = positions[Direction.RIGHT][0] + half_panel
        top = positions[Direction.UP][1] - half_panel
        bottom = positions[Direction.DOWN][1] + half_panel

        left -= panel_expand_px
        right += panel_expand_px
        top -= panel_expand_px
        bottom += panel_expand_px
        
        width = right - left
        height = bottom - top
        
        # Store the overall bounding rect.
        self._panel_rect = pygame.Rect(left, top, width, height)
        
        # Create a full opaque black rectangle for the panel.
        self._panel_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._panel_surface.fill((0, 0, 0, 255))

        # Keep compatibility for callers that expect this attribute.
        self._plus_shape = {
            'vertical': self._panel_rect.copy(),
            'horizontal': self._panel_rect.copy(),
        }
    
    def get_plus_shape(self) -> dict:
        """Get the plus-shape rectangles for forbidden zone calculation"""
        return getattr(self, '_plus_shape', None)
        
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
        Create a single arrow surface with pixel art styling.
        
        The arrow is a solid filled isosceles triangle with pixel art borders:
        - Bounding box: 100×100 px (configurable via size)
        - Triangle: 80px length, 60px base (configurable in ArrowConfig)
        - Pixel art highlight/shadow edges for 3D effect
        
        Args:
            direction: Which way the arrow points
            color: RGB color tuple
            size: Size of the bounding box in pixels
            
        Returns:
            pygame.Surface with the arrow drawn on it
        """
        # Create surface with transparency
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Calculate arrow points using configured dimensions
        points = self._get_arrow_points(direction, size)
        
        # Draw filled arrow (solid triangle)
        pygame.draw.polygon(surface, color, points)
        
        # Pixel art style: chunky border with highlight/shadow
        # Same colors as original, just with pixel art shading
        highlight_color = tuple(min(255, c + 40) for c in color)
        shadow_color = tuple(max(0, c - 40) for c in color)
        
        # Draw chunky pixel art outline (4px thick)
        border_width = 4
        pygame.draw.polygon(surface, shadow_color, points, border_width)
        
        # Add highlight edge based on direction (top-left edges get highlight)
        if len(points) >= 3:
            p0, p1, p2 = points[0], points[1], points[2]
            
            if direction == Direction.UP:
                # Highlight on left edge (p0 to p1) and top portion
                self._draw_pixel_line(surface, p0, p1, highlight_color, 2)
                self._draw_pixel_line(surface, p0, p2, shadow_color, 2)
            elif direction == Direction.DOWN:
                # Highlight on top edges
                self._draw_pixel_line(surface, p1, p2, highlight_color, 2)
                self._draw_pixel_line(surface, p0, p1, shadow_color, 2)
            elif direction == Direction.LEFT:
                # Highlight on top edge
                self._draw_pixel_line(surface, p0, p1, highlight_color, 2)
                self._draw_pixel_line(surface, p0, p2, shadow_color, 2)
            elif direction == Direction.RIGHT:
                # Highlight on top edge
                self._draw_pixel_line(surface, p1, p2, highlight_color, 2)
                self._draw_pixel_line(surface, p0, p2, shadow_color, 2)
        
        return surface
    
    def _draw_pixel_line(self, surface: pygame.Surface, p1: tuple, p2: tuple, 
                         color: Tuple[int, int, int], width: int):
        """Draw a pixel art style line between two points"""
        pygame.draw.line(surface, color, p1, p2, width)
        
    def _get_arrow_points(
        self, 
        direction: Direction, 
        size: int
    ) -> list:
        """
        Calculate polygon points for an arrow shape.
        
        Creates a solid isosceles triangle with:
        - Length (in pointing direction): triangle_length (default 80px)
        - Base width: triangle_base (default 60px)
        
        The triangle is centered in the bounding box.
        
        Args:
            direction: Which way the arrow points
            size: Size of the bounding box
            
        Returns:
            List of (x, y) tuples for polygon vertices
        """
        # Get triangle dimensions from config
        length = self.arrow_config.triangle_length  # 80px default
        base = self.arrow_config.triangle_base      # 60px default
        half_base = base / 2
        
        # Center of the bounding box
        center = size / 2
        
        # Calculate margins to center the triangle
        # For a 100px box with 80px length: margin = (100-80)/2 = 10px
        length_margin = (size - length) / 2
        
        if direction == Direction.UP:
            # Tip at top, base at bottom
            tip_y = length_margin
            base_y = size - length_margin
            return [
                (center, tip_y),                    # Tip (top center)
                (center - half_base, base_y),       # Bottom left
                (center + half_base, base_y),       # Bottom right
            ]
            
        elif direction == Direction.DOWN:
            # Tip at bottom, base at top
            tip_y = size - length_margin
            base_y = length_margin
            return [
                (center, tip_y),                    # Tip (bottom center)
                (center - half_base, base_y),       # Top left
                (center + half_base, base_y),       # Top right
            ]
            
        elif direction == Direction.LEFT:
            # Tip at left, base at right
            tip_x = length_margin
            base_x = size - length_margin
            return [
                (tip_x, center),                    # Tip (left center)
                (base_x, center - half_base),       # Top right
                (base_x, center + half_base),       # Bottom right
            ]
            
        elif direction == Direction.RIGHT:
            # Tip at right, base at left
            tip_x = size - length_margin
            base_x = length_margin
            return [
                (tip_x, center),                    # Tip (right center)
                (base_x, center - half_base),       # Top left
                (base_x, center + half_base),       # Bottom left
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