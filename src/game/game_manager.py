"""
Game Manager
============
Coordinates all game elements and integrates with the BCI system.

Features:
- Level management (generate, reset, advance)
- Player movement from BCI selections
- Score and progress tracking
- Game state management (playing, paused, complete)
- Integration with ArrowManager

This is the main interface between the game and the BCI system.
"""

import pygame
import time
from enum import Enum, auto
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

from config import Config, Direction
from src.game.maze import Maze, MazeConfig, CellType
from src.game.player import Player, PlayerConfig
from src.game.collectible import CollectibleManager, CollectibleConfig
from src.game.renderer import GameRenderer, RenderConfig


class GameState(Enum):
    """Game states"""
    READY = auto()          # Ready to start
    PLAYING = auto()        # Game in progress
    PAUSED = auto()         # Game paused
    LEVEL_COMPLETE = auto() # Level finished
    GAME_OVER = auto()      # Game ended


@dataclass
class GameStats:
    """Statistics for current game session"""
    score: int = 0
    level: int = 1
    moves: int = 0
    items_collected: int = 0
    total_items: int = 0
    time_started: float = 0.0
    time_elapsed: float = 0.0


@dataclass
class GameManagerConfig:
    """Configuration for game manager"""
    # Maze settings per level
    base_maze_width: int = 21
    base_maze_height: int = 15
    maze_growth_per_level: int = 2  # Maze grows each level
    max_maze_width: int = 35
    max_maze_height: int = 25
    
    # Collectibles per level
    base_collectibles: int = 8
    collectibles_per_level: int = 2
    max_collectibles: int = 25
    
    # Cell size (auto-calculated if 0)
    cell_size: int = 0
    
    # Generation mode
    use_corridors: bool = False  # If True, generate simple corridors instead of maze
    
    # Level completion
    require_all_collectibles: bool = True
    require_reach_goal: bool = False  # Optional: also reach goal square
    
    # Timing
    level_complete_delay_ms: float = 2000.0  # Delay before next level


class GameManager:
    """
    Main game coordinator.
    
    Manages:
    - Maze generation and rendering
    - Player movement (from BCI or keyboard)
    - Collectible spawning and collection
    - Level progression
    - Score tracking
    
    Usage:
        manager = GameManager(config)
        manager.initialize(screen_width, screen_height, arrow_panel_rect)
        manager.start_game()
        
        # When BCI selection completes:
        manager.move_player(direction)
        
        # In game loop:
        manager.update(delta_ms)
        manager.draw(screen)
    """
    
    def __init__(self, config: GameManagerConfig = None):
        self.config = config or GameManagerConfig()
        
        # Game elements
        self.maze: Optional[Maze] = None
        self.player: Optional[Player] = None
        self.collectibles: Optional[CollectibleManager] = None
        self.renderer: Optional[GameRenderer] = None
        
        # Screen info
        self._screen_width: int = 0
        self._screen_height: int = 0
        self._arrow_panel_rect: Optional[pygame.Rect] = None
        
        # State
        self._state = GameState.READY
        self._stats = GameStats()
        
        # Level completion
        self._level_complete_timer: float = 0.0
        
        # Callbacks
        self._on_level_complete: Optional[Callable[[int, int], None]] = None
        self._on_game_over: Optional[Callable[[GameStats], None]] = None
        self._on_item_collected: Optional[Callable[[int], None]] = None
        
    def initialize(
        self, 
        screen_width: int, 
        screen_height: int,
        arrow_panel_rect: pygame.Rect = None
    ):
        """
        Initialize game manager.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            arrow_panel_rect: Rectangle of arrow panel (to avoid overlapping)
        """
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._arrow_panel_rect = arrow_panel_rect
        
        # Calculate cell size based on screen
        if self.config.cell_size == 0:
            # Auto-calculate to fit screen with some margin
            max_width = screen_width - 40
            max_height = screen_height - 100  # Leave room for UI
            
            # Use maximum maze size for calculation
            cell_w = max_width // self.config.max_maze_width
            cell_h = max_height // self.config.max_maze_height
            self.config.cell_size = min(cell_w, cell_h, 40)  # Cap at 40px
            
        # Create renderer
        render_config = RenderConfig(cell_size=self.config.cell_size)
        self.renderer = GameRenderer(render_config)
        
        # Create collectible manager
        self.collectibles = CollectibleManager(CollectibleConfig())
        
    def start_game(self):
        """Start a new game from level 1"""
        self._stats = GameStats(
            level=1,
            time_started=time.time()
        )
        self._generate_level()
        self._state = GameState.PLAYING
        
    def restart_level(self):
        """Restart current level"""
        self._generate_level()
        self._state = GameState.PLAYING
        
    def next_level(self):
        """Advance to next level"""
        self._stats.level += 1
        self._generate_level()
        self._state = GameState.PLAYING
        
    def _generate_level(self):
        """Generate maze and place elements for current level"""
        level = self._stats.level
        
        # Calculate maze size for this level
        width = min(
            self.config.base_maze_width + (level - 1) * self.config.maze_growth_per_level,
            self.config.max_maze_width
        )
        height = min(
            self.config.base_maze_height + (level - 1) * self.config.maze_growth_per_level,
            self.config.max_maze_height
        )
        
        # Ensure odd dimensions only for traditional maze mode (not corridors)
        if not self.config.use_corridors:
            if width % 2 == 0:
                width += 1
            if height % 2 == 0:
                height += 1
            
        # Create maze config
        maze_config = MazeConfig(
            width=width,
            height=height,
            cell_size=self.config.cell_size,
            seed=None,  # Random each time
            use_corridors=self.config.use_corridors
        )
        
        self.maze = Maze(maze_config)
        
        # Calculate forbidden zone from arrow panel rect (in grid coordinates)
        # This ensures the arrow panel area acts as an impassable wall
        if self._arrow_panel_rect:
            cell_size = self.config.cell_size
            
            # Calculate maze offset (maze is centered on screen)
            maze_width_px = width * cell_size
            maze_height_px = height * cell_size
            offset_x = (self._screen_width - maze_width_px) // 2
            offset_y = (self._screen_height - maze_height_px) // 2
            
            # Convert arrow panel rect to grid coordinates
            # Use floor for left/top and ceil for right/bottom to ensure full coverage
            # Add padding to create a buffer zone around the panel
            padding = 2  # Extra cells of padding around the arrow panel
            
            # Calculate grid boundaries that fully contain the arrow panel
            # Any cell that even partially overlaps with the panel should be forbidden
            panel_left = self._arrow_panel_rect.left - offset_x
            panel_top = self._arrow_panel_rect.top - offset_y
            panel_right = self._arrow_panel_rect.right - offset_x
            panel_bottom = self._arrow_panel_rect.bottom - offset_y
            
            # Convert to grid coords with proper rounding
            grid_x1 = max(0, int(panel_left // cell_size) - padding)
            grid_y1 = max(0, int(panel_top // cell_size) - padding)
            grid_x2 = min(width, int((panel_right + cell_size - 1) // cell_size) + padding)
            grid_y2 = min(height, int((panel_bottom + cell_size - 1) // cell_size) + padding)
            
            grid_w = grid_x2 - grid_x1
            grid_h = grid_y2 - grid_y1
            
            # Only set forbidden zone if it leaves room for maze around it
            # With corridor mode, we only need 1 cell for paths
            min_corridor_width = 1 if self.config.use_corridors else 2
            if (grid_x1 >= min_corridor_width and 
                width - grid_x2 >= min_corridor_width and
                grid_y1 >= min_corridor_width and 
                height - grid_y2 >= min_corridor_width):
                self.maze.set_forbidden_zone((grid_x1, grid_y1, grid_w, grid_h))
            else:
                print(f"Warning: Arrow panel too large for maze, adjusting forbidden zone")
                print(f"  Maze: {width}x{height}, Calculated zone: ({grid_x1},{grid_y1}) size {grid_w}x{grid_h}")
                print(f"  Corridors: L={grid_x1}, R={width-grid_x2}, T={grid_y1}, B={height-grid_y2}")
                
                # Try with no padding (just cover the panel exactly)
                padding = 0
                grid_x1 = max(0, int(panel_left // cell_size))
                grid_y1 = max(0, int(panel_top // cell_size))
                grid_x2 = min(width, int((panel_right + cell_size - 1) // cell_size))
                grid_y2 = min(height, int((panel_bottom + cell_size - 1) // cell_size))
                grid_w = grid_x2 - grid_x1
                grid_h = grid_y2 - grid_y1
                
                # Set it even if corridors are narrow - corridor mode handles this
                self.maze.set_forbidden_zone((grid_x1, grid_y1, grid_w, grid_h))
                print(f"  Using minimal zone: ({grid_x1},{grid_y1}) size {grid_w}x{grid_h}")
        
        # Generate maze
        self.maze.generate()
        
        # Setup renderer for this maze
        self.renderer.setup(
            self._screen_width, 
            self._screen_height, 
            self.maze,
            self._arrow_panel_rect  # Pass arrow panel rect for hole rendering
        )
        
        # Find valid start position (not in forbidden zone)
        start_pos = self._find_valid_start_position()
        
        # Create player at start position
        player_config = PlayerConfig()
        self.player = Player(player_config, start_pos=start_pos)
        self.player.set_move_callback(self._on_player_move_complete)
        
        # Spawn collectibles
        num_collectibles = min(
            self.config.base_collectibles + (level - 1) * self.config.collectibles_per_level,
            self.config.max_collectibles
        )
        
        self.collectibles.clear()
        
        # Exclude start, goal, and positions too close to start
        exclude_set = {start_pos, self.maze.goal_pos}
        # Also exclude immediate neighbors of start
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            exclude_set.add((start_pos[0] + dx, start_pos[1] + dy))
            
        def get_spawn_position(exclude=None):
            """Get random position for collectible, excluding certain cells"""
            all_exclude = exclude_set.copy()
            if exclude:
                all_exclude.update(exclude)
            return self.maze.get_random_path_cell(all_exclude)
            
        self.collectibles.spawn_random(
            get_spawn_position,
            count=num_collectibles
        )
        
        # Update stats
        self._stats.total_items = self.collectibles.total_count
        self._stats.items_collected = 0
        
    def _find_valid_start_position(self) -> Tuple[int, int]:
        """Find a valid start position that's not in the forbidden zone"""
        # Try the default start position first
        if self.maze.is_walkable(1, 1):
            return (1, 1)
            
        # Otherwise find any walkable cell
        for y in range(1, self.maze.height - 1):
            for x in range(1, self.maze.width - 1):
                if self.maze.is_walkable(x, y):
                    return (x, y)
                    
        # Fallback (shouldn't happen)
        return (1, 1)
        
    def move_player(self, direction: Direction) -> bool:
        """
        Attempt to move player in direction.
        
        Called by BCI system when selection completes.
        
        Args:
            direction: Direction to move
            
        Returns:
            True if move was valid and started
        """
        if self._state != GameState.PLAYING:
            return False
            
        if self.player.is_moving:
            return False
            
        if self.player.can_move(direction, self.maze.is_walkable):
            self.player.move(direction)
            self._stats.moves += 1
            return True
            
        return False
        
    def _on_player_move_complete(self, position: Tuple[int, int]):
        """Called when player finishes moving to a new cell"""
        # Check for collectible
        collected = self.collectibles.collect_at(position[0], position[1])
        if collected:
            self._stats.items_collected = self.collectibles.collected_count
            self._stats.score = self.collectibles.total_score
            
            if self._on_item_collected:
                self._on_item_collected(self.collectibles.get_points(collected.type))
                
        # Check for level completion
        self._check_level_complete()
        
    def _check_level_complete(self):
        """Check if level completion conditions are met"""
        # Check collectibles
        if self.config.require_all_collectibles:
            if not self.collectibles.all_collected():
                return
                
        # Check goal (if required)
        if self.config.require_reach_goal:
            if self.player.grid_position != self.maze.goal_pos:
                return
                
        # Level complete!
        self._state = GameState.LEVEL_COMPLETE
        self._level_complete_timer = 0.0
        
        if self._on_level_complete:
            self._on_level_complete(self._stats.level, self._stats.score)
            
    def update(self, delta_ms: float):
        """
        Update game state.
        
        Args:
            delta_ms: Time elapsed since last update in milliseconds
        """
        if self._state == GameState.PLAYING:
            # Update player animation
            self.player.update(delta_ms)
            
            # Update elapsed time
            self._stats.time_elapsed = time.time() - self._stats.time_started
            
        elif self._state == GameState.LEVEL_COMPLETE:
            # Wait for delay then auto-advance
            self._level_complete_timer += delta_ms
            
            if self._level_complete_timer >= self.config.level_complete_delay_ms:
                self.next_level()
                
    def draw(self, screen: pygame.Surface):
        """
        Draw game elements.
        
        Args:
            screen: Pygame surface to draw on
        """
        if self.maze is None or self.renderer is None:
            return
            
        # Draw maze
        self.renderer.draw_maze(screen, self.maze)
        
        # Draw collectibles
        self.renderer.draw_collectibles(screen, self.collectibles)
        
        # Draw player
        if self.player:
            self.renderer.draw_player(screen, self.player)
            
        # Draw UI
        self.renderer.draw_ui(
            screen,
            score=self._stats.score,
            collected=self._stats.items_collected,
            total=self._stats.total_items,
            level=self._stats.level
        )
        
        # Draw level complete message
        if self._state == GameState.LEVEL_COMPLETE:
            self.renderer.draw_message(
                screen,
                f"Level {self._stats.level} Complete!",
                f"Score: {self._stats.score} | Next level in {int((self.config.level_complete_delay_ms - self._level_complete_timer) / 1000) + 1}s"
            )
            
    def set_callbacks(
        self,
        on_level_complete: Callable[[int, int], None] = None,
        on_game_over: Callable[[GameStats], None] = None,
        on_item_collected: Callable[[int], None] = None,
    ):
        """Set callback functions"""
        self._on_level_complete = on_level_complete
        self._on_game_over = on_game_over
        self._on_item_collected = on_item_collected
        
    @property
    def state(self) -> GameState:
        """Get current game state"""
        return self._state
        
    @property
    def stats(self) -> GameStats:
        """Get current game statistics"""
        return self._stats
        
    @property
    def is_playing(self) -> bool:
        """Whether game is in playing state"""
        return self._state == GameState.PLAYING
        
    @property
    def can_accept_input(self) -> bool:
        """Whether game can accept movement input"""
        return (self._state == GameState.PLAYING and 
                self.player is not None and 
                not self.player.is_moving)
                
    def get_maze_rect(self) -> Optional[pygame.Rect]:
        """Get maze screen rectangle"""
        if self.renderer:
            return self.renderer.get_maze_rect()
        return None


# =============================================================================
# Testing
# =============================================================================

def demo():
    """Demo the game manager"""
    pygame.init()
    
    screen_width, screen_height = 1024, 768
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Game Manager Demo")
    clock = pygame.time.Clock()
    
    # Create game manager
    config = GameManagerConfig(
        base_maze_width=21,
        base_maze_height=15,
        base_collectibles=10,
    )
    
    manager = GameManager(config)
    manager.initialize(screen_width, screen_height)
    
    # Callbacks
    def on_level_complete(level, score):
        print(f"Level {level} complete! Score: {score}")
        
    def on_item_collected(points):
        print(f"Collected item worth {points} points!")
        
    manager.set_callbacks(
        on_level_complete=on_level_complete,
        on_item_collected=on_item_collected,
    )
    
    # Start game
    manager.start_game()
    
    running = True
    font = pygame.font.Font(None, 24)
    
    print("Game Manager Demo")
    print("  Arrow keys: Move player")
    print("  R: Restart level")
    print("  N: Next level")
    print("  ESC: Quit")
    print()
    
    while running:
        delta_ms = clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    manager.restart_level()
                elif event.key == pygame.K_n:
                    manager.next_level()
                    
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
                    
                if direction and manager.can_accept_input:
                    manager.move_player(direction)
                    
        # Update
        manager.update(delta_ms)
        
        # Draw
        screen.fill((20, 20, 25))
        manager.draw(screen)
        
        # Draw controls hint at bottom
        hint = font.render(
            "Arrow keys: Move | R: Restart | N: Next Level | ESC: Quit",
            True, (80, 80, 80)
        )
        screen.blit(hint, (10, screen_height - 25))
        
        pygame.display.flip()
        
    pygame.quit()


if __name__ == "__main__":
    demo()