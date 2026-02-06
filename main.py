#!/usr/bin/env python3
"""
P300 BCI Game - Main Entry Point
================================
Salem State University Capstone Project

A Brain-Computer Interface game using P300 evoked potentials
to control a maze character through flashing arrow stimuli.

Controls:
    SPACE  - Start/Stop BCI selection (arrow flashing)
    S      - Open settings panel (TODO)
    D      - Toggle debug overlay
    ESC    - Quit
    Arrows - Manual movement (for testing)
    1-4    - Simulate BCI selection (Up/Down/Left/Right)

Usage:
    python main.py
    python main.py --fullscreen
    python main.py --debug
    python main.py --width 1280 --height 720
"""

import sys
import argparse
import time

import pygame

from config import Config, Direction
from src.stimulus.arrow_manager import ArrowManager, SelectionState, SelectionResult
from src.ui.settings_panel import SettingsPanel, SettingsValues
from src.game.game_manager import GameManager, GameManagerConfig, GameState
from src.data.session_logger import SessionLogger


# Design resolution - the resolution all game elements are laid out for.
# The game always renders internally at this resolution, then scales to fit
# whatever window size the user requests.
DESIGN_WIDTH = 3072
DESIGN_HEIGHT = 1920


class Application:
    """
    Main application class for P300 BCI Game.
    
    Manages:
    - Pygame initialization and main loop
    - Arrow stimulus system (via ArrowManager)
    - Game state (maze - TODO)
    - User input handling
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.clock = None
        self.screen = None
        
        # Components
        self.arrow_manager: ArrowManager = None
        self.settings_panel: SettingsPanel = None
        self.game_manager: GameManager = None
        self.session_logger: SessionLogger = None
        
        # Resolution-independent rendering
        self.render_surface = None  # Internal surface at design resolution
        self.scale_factor = 1.0
        self.render_offset = (0, 0)
        self.scaled_size = (DESIGN_WIDTH, DESIGN_HEIGHT)
        
        # Display info
        self.num_displays = 1
        self.display_index = 0
        self.display_override = False  # True if manually set via command line
        
        # State
        self.show_debug = config.debug
        self.last_selection: SelectionResult = None
        
        # Fonts (created on initialize)
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        
    def initialize(self):
        """Initialize pygame and all components"""
        pygame.init()
        
        # Detect available displays
        self.num_displays = pygame.display.get_num_displays()
        
        # Auto-select display if not manually overridden
        if not self.display_override:
            # Use second display if available, otherwise use primary
            # This allows the app to work on both single and dual-display setups
            self.display_index = 1 if self.num_displays > 1 else 0
        
        # Create display
        flags = pygame.DOUBLEBUF
        if self.config.display.fullscreen:
            flags |= pygame.FULLSCREEN
            
        self.screen = pygame.display.set_mode(
            (self.config.display.width, self.config.display.height),
            flags,
            display=self.display_index
        )
        pygame.display.set_caption("P300 BCI Game - Salem State University")
        
        self.clock = pygame.time.Clock()
        
        # Create render surface at design resolution
        self.render_surface = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))
        self._calculate_scaling()
        
        # Create fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize arrow manager
        self._init_arrow_manager()
        
        # Initialize session logger
        self._init_session_logger()
        
        # Initialize settings panel
        self._init_settings_panel()
        
        # Initialize game manager
        self._init_game_manager()
        
        # Print startup info
        self._print_startup_info()
        
    def _calculate_scaling(self):
        """Calculate scale factor and offset for resolution-independent rendering.
        
        Maps the design resolution (DESIGN_WIDTH x DESIGN_HEIGHT) to the actual
        window, maintaining aspect ratio with letterboxing if needed.
        """
        window_w = self.config.display.width
        window_h = self.config.display.height
        
        scale_x = window_w / DESIGN_WIDTH
        scale_y = window_h / DESIGN_HEIGHT
        self.scale_factor = min(scale_x, scale_y)
        
        # Calculate scaled dimensions and centering offset (letterboxing)
        scaled_w = int(DESIGN_WIDTH * self.scale_factor)
        scaled_h = int(DESIGN_HEIGHT * self.scale_factor)
        self.scaled_size = (scaled_w, scaled_h)
        self.render_offset = (
            (window_w - scaled_w) // 2,
            (window_h - scaled_h) // 2
        )
        
    def _transform_event(self, event):
        """Transform mouse coordinates from window space to design resolution space."""
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            x, y = event.pos
            # Remove letterbox offset and scale to design resolution
            x = int((x - self.render_offset[0]) / self.scale_factor)
            y = int((y - self.render_offset[1]) / self.scale_factor)
            
            attrs = dict(event.__dict__)
            attrs['pos'] = (x, y)
            if event.type == pygame.MOUSEMOTION and hasattr(event, 'rel'):
                rx, ry = event.rel
                attrs['rel'] = (int(rx / self.scale_factor), int(ry / self.scale_factor))
            return pygame.event.Event(event.type, **attrs)
        return event
        
    def _init_arrow_manager(self):
        """Initialize the arrow stimulus system"""
        self.arrow_manager = ArrowManager(self.config)
        self.arrow_manager.initialize(
            DESIGN_WIDTH, 
            DESIGN_HEIGHT
        )
        
        # Set callbacks
        self.arrow_manager.set_callbacks(
            on_selection_complete=self._on_selection_complete,
            on_state_change=self._on_state_change,
            on_flash_start=self._on_flash_start,
            on_flash_end=self._on_flash_end,
        )
        
    def _init_session_logger(self):
        """Initialize the session logger for data recording"""
        self.session_logger = SessionLogger(output_dir="sessions")
        
    def _init_settings_panel(self):
        """Initialize the settings panel"""
        self.settings_panel = SettingsPanel(
            DESIGN_WIDTH,
            DESIGN_HEIGHT
        )
        
        # Set current values from config
        values = SettingsValues.from_config(self.config)
        self.settings_panel.set_values(values)
        
        # Set callbacks
        self.settings_panel.set_callbacks(
            on_apply=self._on_settings_apply,
            on_cancel=self._on_settings_cancel,
        )
        
    def _init_game_manager(self):
        """Initialize the maze game"""
        # Get arrow panel size to calculate appropriate maze size
        panel_rect = self.arrow_manager.get_panel_rect()
        
        # =================================================================
        # MAZE SIZE CONFIGURATION
        # =================================================================
        # Cell size determines how large each maze cell appears on screen.
        # Larger cells = fewer cells = simpler maze (easier to play)
        # Smaller cells = more cells = complex maze (harder to play)
        #
        # With 192px cells and the arrow panel (~1100×1160), we use
        # CORRIDOR MODE instead of complex maze generation. This creates
        # simple paths around the arrow panel that are easy to navigate
        # with BCI control.
        #
        # To adjust: change cell_size below
        # =================================================================
        
        cell_size = 160  # <-- CHANGE THIS TO ADJUST MAZE SIZE
        
        # Calculate maze dimensions to fill screen
        margin = 0  # No margin - fill entire screen
        
        maze_width_cells = (DESIGN_WIDTH - margin * 2) // cell_size
        maze_height_cells = (DESIGN_HEIGHT - margin * 2) // cell_size
        
        # For corridor mode, we don't need odd dimensions
        # Use corridor mode for large cells (simpler paths)
        use_corridors = cell_size >= 128
        
        game_config = GameManagerConfig(
            base_maze_width=maze_width_cells,
            base_maze_height=maze_height_cells,
            max_maze_width=maze_width_cells,  # Don't grow beyond screen
            max_maze_height=maze_height_cells,
            maze_growth_per_level=0,  # Keep same size, just regenerate
            base_collectibles=8,  # Fewer collectibles for simpler corridor layout
            collectibles_per_level=2,
            cell_size=cell_size,
            use_corridors=use_corridors,  # Use corridor mode for large cells
        )
        
        self.game_manager = GameManager(game_config)
        
        # Get arrow positions for plus-shaped forbidden zone
        arrow_positions = self.config.layout.get_positions(
            DESIGN_WIDTH,
            DESIGN_HEIGHT
        )
        
        self.game_manager.initialize(
            DESIGN_WIDTH,
            DESIGN_HEIGHT,
            panel_rect,
            arrow_positions
        )
        
        # Set callbacks
        self.game_manager.set_callbacks(
            on_level_complete=self._on_level_complete,
            on_item_collected=self._on_item_collected,
        )
        
        # Start the game
        self.game_manager.start_game()
        
    def _print_startup_info(self):
        """Print configuration info to console"""
        print()
        print("=" * 50)
        print("P300 BCI Game - Salem State University")
        print("=" * 50)
        print()
        print("Display:")
        print(f"  Available displays: {self.num_displays}")
        display_type = "(primary)" if self.display_index == 0 else "(secondary)"
        override_text = " [manual]" if self.display_override else " [auto]"
        print(f"  Using display: {self.display_index + 1} {display_type}{override_text}")
        print()
        print("Configuration:")
        print(f"  Window: {self.config.display.width}x{self.config.display.height}")
        print(f"  Design: {DESIGN_WIDTH}x{DESIGN_HEIGHT}")
        print(f"  Scale: {self.scale_factor:.3f}x")
        print(f"  Fullscreen: {self.config.display.fullscreen}")
        print()
        print("Timing:")
        print(f"  Flash duration: {self.config.timing.flash_duration_ms}ms")
        print(f"  ISI: {self.config.timing.isi_ms}ms")
        print(f"  SOA: {self.config.timing.soa_ms}ms")
        print(f"  Flash rate: {self.config.timing.flash_rate_hz:.1f}Hz per arrow")
        print(f"  Sequences: {self.config.timing.num_sequences}")
        print()
        print("Controls:")
        print("  SPACE  - Start/Stop BCI selection")
        print("  S      - Open settings panel")
        print("  D      - Toggle debug info")
        print("  R      - Restart current level")
        print("  N      - Skip to next level")
        print("  1-4    - Simulate BCI selection (Up/Down/Left/Right)")
        print("  Arrows - Manual movement (testing)")
        print("  ESC    - Quit")
        print()
        print("=" * 50)
        print()
        
    def run(self):
        """Main game loop"""
        self.running = True
        
        while self.running:
            # Handle events
            self._handle_events()
            
            # Update
            self._update()
            
            # Draw
            self._draw()
            
            # Cap framerate
            self.clock.tick(self.config.display.fps)
            
        self._cleanup()
        
    def _handle_events(self):
        """Process input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            
            # Transform mouse coordinates from window space to design space
            event = self._transform_event(event)
                
            # Let settings panel handle events first (if visible)
            if self.settings_panel and self.settings_panel.is_visible:
                if self.settings_panel.handle_event(event):
                    continue  # Event consumed by panel
                    
            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
                
    def _handle_keydown(self, key: int):
        """Handle keyboard input"""
        # Quit
        if key == pygame.K_ESCAPE:
            self.running = False
            
        # Toggle BCI selection
        elif key == pygame.K_SPACE:
            self._toggle_selection()
            
        # Toggle debug
        elif key == pygame.K_d:
            self.show_debug = not self.show_debug
            
        # Toggle settings panel
        elif key == pygame.K_s:
            self._toggle_settings()
            
        # Game controls
        elif key == pygame.K_r:
            # Restart level
            if self.game_manager:
                self.game_manager.restart_level()
                print("Level restarted")
        elif key == pygame.K_n:
            # Next level (for testing)
            if self.game_manager:
                self.game_manager.next_level()
                print(f"Advanced to level {self.game_manager.stats.level}")
            
        # Simulate selections (for testing)
        elif key == pygame.K_1:
            self._simulate_selection(Direction.UP)
        elif key == pygame.K_2:
            self._simulate_selection(Direction.DOWN)
        elif key == pygame.K_3:
            self._simulate_selection(Direction.LEFT)
        elif key == pygame.K_4:
            self._simulate_selection(Direction.RIGHT)
            
        # Manual movement (for testing game without BCI)
        elif key == pygame.K_UP:
            self._manual_move(Direction.UP)
        elif key == pygame.K_DOWN:
            self._manual_move(Direction.DOWN)
        elif key == pygame.K_LEFT:
            self._manual_move(Direction.LEFT)
        elif key == pygame.K_RIGHT:
            self._manual_move(Direction.RIGHT)
            
    def _toggle_selection(self):
        """Start or stop BCI selection"""
        if self.arrow_manager.is_active:
            self.arrow_manager.stop_selection()
            # Cancel session if stopped early
            if self.session_logger and self.session_logger.is_active:
                self.session_logger.cancel_session()
            print("Selection stopped")
        else:
            # Start session logging before starting selection
            if self.session_logger:
                self.session_logger.start_session(
                    flash_duration_ms=self.config.timing.flash_duration_ms,
                    isi_ms=self.config.timing.isi_ms,
                    num_sequences=self.config.timing.num_sequences,
                    inter_sequence_pause_ms=self.config.timing.inter_sequence_pause_ms,
                    flash_pattern=self.config.timing.flash_pattern.name,
                    color_scheme=self.config.arrows.color_scheme.name,
                )
            self.arrow_manager.start_selection()
            print("Selection started - arrows flashing")
            
    def _simulate_selection(self, direction: Direction):
        """Simulate a classifier result (for testing)"""
        if self.arrow_manager.state in (SelectionState.FLASHING, SelectionState.PROCESSING):
            self.arrow_manager.simulate_selection(direction)
            print(f"Simulated selection: {direction.value}")
        else:
            print("Start selection first (SPACE)")
            
    def _manual_move(self, direction: Direction):
        """Handle manual movement (for testing game without BCI)"""
        if self.game_manager and self.game_manager.can_accept_input:
            moved = self.game_manager.move_player(direction)
            if moved:
                print(f"Manual move: {direction.value}")
            else:
                print(f"Manual move: {direction.value} (blocked)")
        
    def _toggle_settings(self):
        """Toggle the settings panel"""
        if self.settings_panel.is_visible:
            self.settings_panel.hide()
        else:
            # Don't open settings during active selection
            if self.arrow_manager.is_active:
                print("Cannot open settings during active selection")
                return
                
            # Update panel with current values
            values = SettingsValues.from_config(self.config)
            self.settings_panel.set_values(values)
            self.settings_panel.show()
            print("Settings panel opened")
            
    def _on_settings_apply(self, values: SettingsValues):
        """Called when settings are applied"""
        print(f"Settings applied:")
        print(f"  Flash: {values.flash_duration_ms}ms")
        print(f"  ISI: {values.isi_ms}ms")
        print(f"  Sequences: {values.num_sequences}")
        print(f"  Color: {values.color_scheme.name}")
        print(f"  SOA: {values.soa_ms}ms ({values.flash_rate_hz:.1f}Hz)")
        
        # Apply to config
        values.apply_to_config(self.config)
        
        # Reinitialize arrow manager with new settings
        self.arrow_manager.shutdown()
        self._init_arrow_manager()
        
    def _on_settings_cancel(self):
        """Called when settings are cancelled"""
        print("Settings cancelled")
        
    def _on_selection_complete(self, result: SelectionResult):
        """Called when BCI selection completes"""
        self.last_selection = result
        
        # End and save session
        if self.session_logger and self.session_logger.is_active:
            filepath = self.session_logger.end_session(
                selected_direction=result.direction
            )
            if filepath:
                print(f"Session saved: {filepath}")
        
        if result.direction:
            print(f"Selection: {result.direction.value} "
                  f"({result.duration_ms:.0f}ms, "
                  f"timing OK: {result.timing_stats['acceptable']})")
            
            # Move player in game
            if self.game_manager and self.game_manager.can_accept_input:
                moved = self.game_manager.move_player(result.direction)
                if not moved:
                    print("  (blocked by wall)")
        else:
            print("Selection: None (timeout or cancelled)")
            
    def _on_flash_start(self, direction: Direction, sequence: int, timestamp_ms: float):
        """Called when a flash begins - log to session"""
        if self.session_logger and self.session_logger.is_active:
            self.session_logger.log_flash_start(direction, sequence, timestamp_ms)
            
    def _on_flash_end(self, direction: Direction, sequence: int, timestamp_ms: float):
        """Called when a flash ends - log to session"""
        if self.session_logger and self.session_logger.is_active:
            self.session_logger.log_flash_end(direction, sequence, timestamp_ms)
            
    def _on_level_complete(self, level: int, score: int):
        """Called when a game level is completed"""
        print(f"Level {level} complete! Score: {score}")
        
    def _on_item_collected(self, points: int):
        """Called when player collects an item"""
        if self.show_debug:
            print(f"  Collected item: +{points} points")
            
    def _on_state_change(self, state: SelectionState):
        """Called when selection state changes"""
        if self.show_debug:
            print(f"  State -> {state.name}")
            
    def _update(self):
        """Update game state"""
        delta_ms = self.clock.get_time()
        
        # Update arrow manager
        self.arrow_manager.update()
        
        # Update game manager
        if self.game_manager:
            self.game_manager.update(delta_ms)
        
    def _draw(self):
        """Render frame.
        
        All components render to self.render_surface at the design resolution
        (DESIGN_WIDTH x DESIGN_HEIGHT). The result is then scaled to fit the
        actual window, maintaining aspect ratio.
        """
        # Clear render surface
        self.render_surface.fill(self.config.display.background_color)
        
        # Draw game (maze, collectibles, player) - behind arrows
        if self.game_manager:
            self.game_manager.draw(self.render_surface)
        
        # Draw arrows (on top of game)
        self.arrow_manager.draw(self.render_surface)
        
        # Draw UI overlays (debug only now - scoreboard is part of game renderer)
        if self.show_debug:
            self._draw_debug()
            
        # Draw settings panel (on top of everything)
        if self.settings_panel:
            self.settings_panel.draw(self.render_surface)
        
        # Scale render surface to actual window
        self.screen.fill((0, 0, 0))  # Black letterbox bars
        if self.scale_factor == 1.0 and self.render_offset == (0, 0):
            # No scaling needed - direct blit
            self.screen.blit(self.render_surface, (0, 0))
        else:
            scaled = pygame.transform.smoothscale(self.render_surface, self.scaled_size)
            self.screen.blit(scaled, self.render_offset)
            
        # Flip display
        pygame.display.flip()
        
    def _draw_status(self):
        """Draw status bar at bottom"""
        # Background bar
        bar_height = 40
        bar_rect = pygame.Rect(
            0, 
            DESIGN_HEIGHT - bar_height,
            DESIGN_WIDTH,
            bar_height
        )
        pygame.draw.rect(self.render_surface, (15, 15, 15), bar_rect)
        
        # Status text
        state = self.arrow_manager.state
        if state == SelectionState.IDLE:
            status = "Press SPACE to start BCI selection"
            color = (50, 50, 50)
        elif state == SelectionState.FLASHING:
            progress = self.arrow_manager.progress * 100
            status = f"Flashing... {progress:.0f}%"
            color = (45, 70, 45)
        elif state == SelectionState.PROCESSING:
            status = "Processing... (Press 1-4 to simulate)"
            color = (70, 70, 45)
        elif state == SelectionState.FEEDBACK:
            status = f"Selected: {self.last_selection.direction.value if self.last_selection else '?'}"
            color = (45, 60, 80)
        else:
            status = str(state.name)
            color = (50, 50, 50)
            
        text = self.font_medium.render(status, True, color)
        text_rect = text.get_rect(
            centerx=DESIGN_WIDTH // 2,
            centery=DESIGN_HEIGHT - bar_height // 2
        )
        self.render_surface.blit(text, text_rect)
        
    def _draw_debug(self):
        """Draw debug information overlay"""
        lines = [
            f"FPS: {self.clock.get_fps():.1f}",
            f"State: {self.arrow_manager.state.name}",
            f"Progress: {self.arrow_manager.progress * 100:.0f}%",
            "",
            f"Flash: {self.config.timing.flash_duration_ms}ms",
            f"ISI: {self.config.timing.isi_ms}ms",
            f"Sequences: {self.config.timing.num_sequences}",
            "",
            f"Window: {self.config.display.width}x{self.config.display.height}",
            f"Scale: {self.scale_factor:.2f}x",
        ]
        
        if self.last_selection:
            lines.extend([
                "",
                f"Last: {self.last_selection.direction.value if self.last_selection.direction else 'None'}",
                f"Time: {self.last_selection.duration_ms:.0f}ms",
            ])
            
        # Draw background
        padding = 10
        line_height = 20
        width = 180
        height = len(lines) * line_height + padding * 2
        
        bg_rect = pygame.Rect(
            DESIGN_WIDTH - width - padding,
            padding,
            width,
            height
        )
        bg_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 180))
        self.render_surface.blit(bg_surface, bg_rect.topleft)
        
        # Draw text
        y = padding * 2
        for line in lines:
            if line:
                text = self.font_small.render(line, True, (60, 60, 60))
                self.render_surface.blit(text, 
                    (DESIGN_WIDTH - width, y))
            y += line_height
            
    def _cleanup(self):
        """Clean up resources"""
        if self.arrow_manager:
            self.arrow_manager.shutdown()
        pygame.quit()
        print()
        print("Application closed.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="P300 BCI Game - Salem State University Capstone"
    )
    parser.add_argument(
        "--fullscreen", "-f", 
        action="store_true",
        help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--debug", "-d", 
        action="store_true",
        help="Enable debug overlay"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=3072,
        help="Window width (default: 3072)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=1920,
        help="Window height (default: 1920)"
    )
    parser.add_argument(
        "--sequences", 
        type=int, 
        default=None,
        help="Number of sequences per selection (default: from config)"
    )
    parser.add_argument(
        "--display",
        type=int,
        default=None,
        help="Display index to use (0=primary, 1=secondary, etc.). If not specified, automatically uses second display when available"
    )
    return parser.parse_args()


def main():
    """Entry point"""
    args = parse_args()
    
    # Create configuration
    config = Config()
    config.display.width = args.width
    config.display.height = args.height
    config.display.fullscreen = args.fullscreen
    config.debug = args.debug
    
    if args.sequences:
        config.timing.num_sequences = args.sequences
    
    # Create and run application
    try:
        app = Application(config)
        
        # Allow manual display override via command line
        if args.display is not None:
            app.display_index = args.display
            app.display_override = True
            
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    
    return 0


if __name__ == "__main__":
    sys.exit(main())