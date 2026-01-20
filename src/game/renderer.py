"""
Game Renderer
=============
Renders game elements using simple shapes.

Designed for easy upgrading to sprites/textures later:
- All rendering is centralized in this class
- Each element type has its own draw method
- Colors and sizes are configurable
- Supports "hole" in maze for arrow panel

Currently renders:
- Maze walls and paths as rectangles
- Player as a circle with direction indicator
- Collectibles as circles/diamonds with animation
- UI elements (score, status)
"""

import pygame
import math
import time
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from src.game.maze import Maze, CellType
from src.game.player import Player
from src.game.collectible import CollectibleManager, Collectible, CollectibleType


@dataclass
class RenderConfig:
    """Rendering configuration"""
    # Cell size
    cell_size: int = 32
    
    # Maze colors (grayscale to not interfere with arrows)
    wall_color: Tuple[int, int, int] = (50, 50, 55)
    path_color: Tuple[int, int, int] = (30, 30, 35)
    start_color: Tuple[int, int, int] = (40, 60, 40)
    goal_color: Tuple[int, int, int] = (60, 40, 60)
    
    # Player colors
    player_color: Tuple[int, int, int] = (100, 180, 100)
    player_outline: Tuple[int, int, int] = (60, 140, 60)
    player_direction_color: Tuple[int, int, int] = (150, 220, 150)
    
    # Collectible colors
    coin_color: Tuple[int, int, int] = (200, 180, 50)
    gem_color: Tuple[int, int, int] = (50, 180, 200)
    star_color: Tuple[int, int, int] = (220, 200, 80)
    
    # UI colors
    ui_bg_color: Tuple[int, int, int] = (20, 20, 25)
    ui_text_color: Tuple[int, int, int] = (180, 180, 190)
    ui_highlight_color: Tuple[int, int, int] = (100, 180, 100)
    
    # Sizes
    player_size_ratio: float = 0.65   # Player size relative to cell
    collectible_size_ratio: float = 0.4  # Collectible size relative to cell


class GameRenderer:
    """
    Renders all game elements.
    
    Supports a "hole" in the middle of the maze where the arrow panel sits.
    The maze wraps around this hole.
    
    Usage:
        renderer = GameRenderer(config)
        renderer.setup(screen_width, screen_height, maze, arrow_panel_rect)
        
        # In game loop:
        renderer.draw_maze(screen, maze)
        renderer.draw_collectibles(screen, collectible_manager)
        renderer.draw_player(screen, player)
        renderer.draw_ui(screen, score, level)
    """
    
    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
        
        # Calculated values (set in setup)
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._maze_width_px: int = 0
        self._maze_height_px: int = 0
        
        # Arrow panel hole (in grid coordinates)
        self._hole_rect: Optional[pygame.Rect] = None  # Screen coords
        self._hole_grid_rect: Optional[pygame.Rect] = None  # Grid coords
        
        # Cached surfaces for performance
        self._maze_surface: Optional[pygame.Surface] = None
        self._maze_dirty: bool = True
        
        # Fonts
        self._font_large: Optional[pygame.font.Font] = None
        self._font_medium: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None
        
        # Animation time
        self._start_time: float = time.time()
        
    def setup(
        self, 
        screen_width: int, 
        screen_height: int, 
        maze: Maze,
        arrow_panel_rect: pygame.Rect = None
    ):
        """
        Setup renderer for given screen and maze.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            maze: Maze to render
            arrow_panel_rect: Rectangle of arrow panel (hole in maze)
        """
        cell_size = self.config.cell_size
        
        # Calculate maze dimensions
        self._maze_width_px = maze.width * cell_size
        self._maze_height_px = maze.height * cell_size
        
        # Center maze on screen
        self._offset_x = (screen_width - self._maze_width_px) // 2
        self._offset_y = (screen_height - self._maze_height_px) // 2
        
        # Store hole rect and convert to grid coordinates
        self._hole_rect = arrow_panel_rect
        if arrow_panel_rect:
            # Convert screen coords to grid coords (with some padding)
            padding = 1  # Extra cell padding around hole
            grid_x1 = max(0, (arrow_panel_rect.left - self._offset_x) // cell_size - padding)
            grid_y1 = max(0, (arrow_panel_rect.top - self._offset_y) // cell_size - padding)
            grid_x2 = min(maze.width, (arrow_panel_rect.right - self._offset_x) // cell_size + padding + 1)
            grid_y2 = min(maze.height, (arrow_panel_rect.bottom - self._offset_y) // cell_size + padding + 1)
            
            self._hole_grid_rect = pygame.Rect(
                grid_x1, grid_y1,
                grid_x2 - grid_x1, grid_y2 - grid_y1
            )
        else:
            self._hole_grid_rect = None
        
        # Initialize fonts
        self._font_large = pygame.font.Font(None, 36)
        self._font_medium = pygame.font.Font(None, 28)
        self._font_small = pygame.font.Font(None, 22)
        
        # Create maze surface cache
        self._maze_surface = pygame.Surface(
            (self._maze_width_px, self._maze_height_px),
            pygame.SRCALPHA  # Support transparency for hole
        )
        self._maze_dirty = True
        
    def grid_to_screen(self, grid_x: float, grid_y: float) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates"""
        cell_size = self.config.cell_size
        screen_x = int(self._offset_x + grid_x * cell_size + cell_size // 2)
        screen_y = int(self._offset_y + grid_y * cell_size + cell_size // 2)
        return (screen_x, screen_y)
        
    def is_in_hole(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid position is inside the arrow panel hole"""
        if self._hole_grid_rect is None:
            return False
        return self._hole_grid_rect.collidepoint(grid_x, grid_y)
        
    def invalidate_maze_cache(self):
        """Mark maze cache as dirty (call when maze changes)"""
        self._maze_dirty = True
        
    def draw_maze(self, screen: pygame.Surface, maze: Maze):
        """
        Draw the maze with hole for arrow panel.
        
        Uses caching for performance since maze rarely changes.
        """
        # Rebuild cache if needed
        if self._maze_dirty:
            self._render_maze_to_cache(maze)
            self._maze_dirty = False
            
        # Blit cached surface
        screen.blit(self._maze_surface, (self._offset_x, self._offset_y))
        
    def _render_maze_to_cache(self, maze: Maze):
        """Render maze to cached surface, leaving hole transparent"""
        cell_size = self.config.cell_size
        
        # Clear with transparency
        self._maze_surface.fill((0, 0, 0, 0))
        
        for y in range(maze.height):
            for x in range(maze.width):
                # Skip cells in the hole area
                if self.is_in_hole(x, y):
                    continue
                    
                cell = maze.get_cell(x, y)
                
                rect = pygame.Rect(
                    x * cell_size,
                    y * cell_size,
                    cell_size,
                    cell_size
                )
                
                if cell == CellType.WALL:
                    pygame.draw.rect(self._maze_surface, self.config.wall_color, rect)
                elif cell == CellType.START:
                    pygame.draw.rect(self._maze_surface, self.config.start_color, rect)
                elif cell == CellType.GOAL:
                    pygame.draw.rect(self._maze_surface, self.config.goal_color, rect)
                else:  # PATH
                    pygame.draw.rect(self._maze_surface, self.config.path_color, rect)
                    
    def draw_player(self, screen: pygame.Surface, player: Player):
        """Draw the player (skip if in hole area)"""
        cell_size = self.config.cell_size
        size = int(cell_size * self.config.player_size_ratio)
        
        # Get interpolated position
        grid_x, grid_y = player.render_position
        
        # Skip if in hole
        if self.is_in_hole(int(grid_x), int(grid_y)):
            return
            
        screen_x, screen_y = self.grid_to_screen(grid_x, grid_y)
        
        # Draw player body (circle)
        pygame.draw.circle(
            screen,
            self.config.player_color,
            (screen_x, screen_y),
            size // 2
        )
        
        # Draw outline
        pygame.draw.circle(
            screen,
            self.config.player_outline,
            (screen_x, screen_y),
            size // 2,
            2
        )
        
        # Draw direction indicator
        direction = player.current_direction
        if direction:
            self._draw_direction_indicator(
                screen, screen_x, screen_y, direction, size // 2
            )
            
    def _draw_direction_indicator(
        self, 
        screen: pygame.Surface,
        cx: int, cy: int, 
        direction, 
        radius: int
    ):
        """Draw a small indicator showing movement direction"""
        from config import Direction
        
        # Direction to angle (in radians)
        angles = {
            Direction.UP: -math.pi / 2,
            Direction.DOWN: math.pi / 2,
            Direction.LEFT: math.pi,
            Direction.RIGHT: 0,
        }
        angle = angles.get(direction, 0)
        
        # Calculate indicator position
        indicator_dist = radius * 0.6
        ix = int(cx + math.cos(angle) * indicator_dist)
        iy = int(cy + math.sin(angle) * indicator_dist)
        
        # Draw small circle
        pygame.draw.circle(
            screen,
            self.config.player_direction_color,
            (ix, iy),
            radius // 4
        )
        
    def draw_collectibles(self, screen: pygame.Surface, manager: CollectibleManager):
        """Draw all collectibles (skip those in hole)"""
        current_time = time.time()
        
        for item in manager.active_collectibles:
            # Skip if in hole
            if self.is_in_hole(item.x, item.y):
                continue
            self._draw_collectible(screen, item, manager.config, current_time)
            
    def _draw_collectible(
        self, 
        screen: pygame.Surface, 
        item: Collectible,
        config,
        current_time: float
    ):
        """Draw a single collectible"""
        cell_size = self.config.cell_size
        size = int(cell_size * self.config.collectible_size_ratio)
        
        # Get position with animation offset
        offset = item.get_animation_offset(current_time, config)
        screen_x, screen_y = self.grid_to_screen(item.x, item.y)
        screen_y += int(offset * cell_size)
        
        # Draw based on type
        if item.type == CollectibleType.COIN:
            self._draw_coin(screen, screen_x, screen_y, size)
        elif item.type == CollectibleType.GEM:
            self._draw_gem(screen, screen_x, screen_y, size, current_time, config)
        elif item.type == CollectibleType.STAR:
            self._draw_star(screen, screen_x, screen_y, size, current_time, config)
        else:
            # Default: simple circle
            pygame.draw.circle(screen, (200, 200, 200), (screen_x, screen_y), size // 2)
            
    def _draw_coin(self, screen: pygame.Surface, cx: int, cy: int, size: int):
        """Draw a coin (circle with inner circle)"""
        color = self.config.coin_color
        
        # Outer circle
        pygame.draw.circle(screen, color, (cx, cy), size // 2)
        
        # Inner circle (darker)
        inner_color = tuple(max(0, c - 50) for c in color)
        pygame.draw.circle(screen, inner_color, (cx, cy), size // 3)
        
    def _draw_gem(
        self, screen: pygame.Surface, 
        cx: int, cy: int, size: int,
        current_time: float, config
    ):
        """Draw a gem (diamond shape)"""
        color = self.config.gem_color
        
        # Diamond points
        half = size // 2
        points = [
            (cx, cy - half),      # Top
            (cx + half, cy),      # Right
            (cx, cy + half),      # Bottom
            (cx - half, cy),      # Left
        ]
        
        pygame.draw.polygon(screen, color, points)
        
        # Outline
        outline_color = tuple(min(255, c + 50) for c in color)
        pygame.draw.polygon(screen, outline_color, points, 2)
        
    def _draw_star(
        self, screen: pygame.Surface, 
        cx: int, cy: int, size: int,
        current_time: float, config
    ):
        """Draw a star (5-pointed)"""
        color = self.config.star_color
        
        # Rotation based on time
        rotation = (current_time * config.rotation_speed) % 360
        rotation = math.radians(rotation)
        
        # Generate star points
        points = []
        outer_radius = size // 2
        inner_radius = size // 4
        
        for i in range(10):
            angle = rotation + (i * math.pi / 5) - (math.pi / 2)
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = cx + int(math.cos(angle) * radius)
            y = cy + int(math.sin(angle) * radius)
            points.append((x, y))
            
        pygame.draw.polygon(screen, color, points)
        
    def draw_ui(
        self, 
        screen: pygame.Surface, 
        score: int,
        collected: int,
        total: int,
        level: int = 1,
        rect: pygame.Rect = None
    ):
        """
        Draw game UI overlay.
        
        Args:
            screen: Surface to draw on
            score: Current score
            collected: Number of items collected
            total: Total number of items
            level: Current level number
            rect: Optional rect to avoid (e.g., arrow panel)
        """
        # Draw in top-left corner
        x, y = 15, 15
        
        # Background panel
        panel_width = 180
        panel_height = 80
        panel_rect = pygame.Rect(x - 10, y - 10, panel_width, panel_height)
        
        # Semi-transparent background
        bg_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        bg_surface.fill((*self.config.ui_bg_color, 200))
        screen.blit(bg_surface, panel_rect.topleft)
        
        # Border
        pygame.draw.rect(screen, (60, 60, 70), panel_rect, 1, border_radius=5)
        
        # Score
        score_text = self._font_medium.render(f"Score: {score}", True, self.config.ui_highlight_color)
        screen.blit(score_text, (x, y))
        
        # Items collected
        items_text = self._font_small.render(
            f"Items: {collected}/{total}", 
            True, 
            self.config.ui_text_color
        )
        screen.blit(items_text, (x, y + 28))
        
        # Level
        level_text = self._font_small.render(f"Level: {level}", True, self.config.ui_text_color)
        screen.blit(level_text, (x, y + 48))
        
    def draw_message(
        self, 
        screen: pygame.Surface, 
        message: str,
        sub_message: str = "",
        color: Tuple[int, int, int] = None
    ):
        """
        Draw a centered message (e.g., "Level Complete!")
        
        Args:
            screen: Surface to draw on
            message: Main message text
            sub_message: Smaller text below
            color: Text color (default: highlight color)
        """
        color = color or self.config.ui_highlight_color
        
        # Get screen center
        screen_rect = screen.get_rect()
        cx, cy = screen_rect.center
        
        # Draw semi-transparent backdrop
        backdrop_width = 400
        backdrop_height = 100
        backdrop = pygame.Surface((backdrop_width, backdrop_height), pygame.SRCALPHA)
        backdrop.fill((0, 0, 0, 180))
        backdrop_rect = backdrop.get_rect(center=(cx, cy))
        screen.blit(backdrop, backdrop_rect)
        
        # Draw main message
        text = self._font_large.render(message, True, color)
        text_rect = text.get_rect(center=(cx, cy - 15))
        screen.blit(text, text_rect)
        
        # Draw sub-message
        if sub_message:
            sub_text = self._font_small.render(sub_message, True, self.config.ui_text_color)
            sub_rect = sub_text.get_rect(center=(cx, cy + 20))
            screen.blit(sub_text, sub_rect)
            
    def get_maze_rect(self) -> pygame.Rect:
        """Get the screen rectangle occupied by the maze"""
        return pygame.Rect(
            self._offset_x,
            self._offset_y,
            self._maze_width_px,
            self._maze_height_px
        )
        
    def get_hole_grid_rect(self) -> Optional[pygame.Rect]:
        """Get the grid rectangle of the hole (for maze generation)"""
        return self._hole_grid_rect


# =============================================================================
# Testing
# =============================================================================

def demo():
    """Visual demo of the renderer"""
    pygame.init()
    
    from src.game.maze import Maze, MazeConfig
    from src.game.player import Player, PlayerConfig
    from src.game.collectible import CollectibleManager, CollectibleConfig
    from config import Direction
    
    # Setup
    screen_width, screen_height = 1024, 768
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Game Renderer Demo")
    clock = pygame.time.Clock()
    
    # Simulate arrow panel in center
    arrow_panel_rect = pygame.Rect(
        screen_width // 2 - 200,
        screen_height // 2 - 150,
        400, 300
    )
    
    # Create game elements
    maze_config = MazeConfig(width=31, height=23, cell_size=32, seed=42)
    maze = Maze(maze_config)
    maze.generate()
    
    player_config = PlayerConfig()
    player = Player(player_config, start_pos=maze.start_pos)
    
    collectible_config = CollectibleConfig()
    collectibles = CollectibleManager(collectible_config)
    collectibles.spawn_random(
        lambda exclude=None: maze.get_random_path_cell(exclude),
        count=15
    )
    
    # Create renderer
    render_config = RenderConfig(cell_size=32)
    renderer = GameRenderer(render_config)
    renderer.setup(screen_width, screen_height, maze, arrow_panel_rect)
    
    running = True
    score = 0
    
    print("Renderer Demo (with hole)")
    print("  Arrow keys: Move player")
    print("  R: Regenerate maze")
    print("  ESC: Quit")
    
    while running:
        delta_ms = clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Regenerate
                    maze.generate()
                    renderer.invalidate_maze_cache()
                    player.set_position(*maze.start_pos)
                    collectibles.clear()
                    collectibles.spawn_random(
                        lambda exclude=None: maze.get_random_path_cell(exclude),
                        count=15
                    )
                    score = 0
                    
                # Movement
                direction = None
                if event.key == pygame.K_UP:
                    direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    direction = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    direction = Direction.RIGHT
                    
                if direction and player.can_move(direction, maze.is_walkable):
                    player.move(direction)
                    
        # Update
        player.update(delta_ms)
        
        # Check collectible pickup
        px, py = player.grid_position
        collected = collectibles.collect_at(px, py)
        if collected:
            score = collectibles.total_score
            
        # Draw
        screen.fill((20, 20, 25))
        renderer.draw_maze(screen, maze)
        renderer.draw_collectibles(screen, collectibles)
        renderer.draw_player(screen, player)
        
        # Draw placeholder for arrow panel
        pygame.draw.rect(screen, (40, 40, 50), arrow_panel_rect)
        pygame.draw.rect(screen, (80, 80, 90), arrow_panel_rect, 2)
        
        renderer.draw_ui(
            screen, 
            score=score,
            collected=collectibles.collected_count,
            total=collectibles.total_count
        )
        
        # Win message
        if collectibles.all_collected():
            renderer.draw_message(
                screen, 
                "Level Complete!",
                "Press R to play again"
            )
            
        pygame.display.flip()
        
    pygame.quit()
    

if __name__ == "__main__":
    demo()