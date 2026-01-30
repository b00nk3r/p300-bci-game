"""
Settings Panel
==============
Runtime configuration panel for P300 BCI parameters.

Features:
- Sliders for timing parameters (flash duration, ISI, sequences)
- Dropdown for color scheme selection
- Real-time SOA calculation display
- Apply/Cancel buttons
- Keyboard shortcuts (Enter to apply, Escape to cancel)

All UI elements are custom pygame implementations (no external dependencies).
"""

import pygame
from typing import Callable, Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from config import Config, ColorScheme, FlashPattern


@dataclass
class SettingsValues:
    """Current settings values that can be modified"""
    flash_duration_ms: int = 100
    isi_ms: int = 125
    num_sequences: int = 10
    color_scheme: ColorScheme = ColorScheme.GRAY_WHITE
    
    @property
    def soa_ms(self) -> int:
        """Stimulus Onset Asynchrony"""
        return self.flash_duration_ms + self.isi_ms
    
    @property
    def flash_rate_hz(self) -> float:
        """Flash rate per arrow"""
        return 1000 / self.soa_ms if self.soa_ms > 0 else 0
    
    @classmethod
    def from_config(cls, config: Config) -> 'SettingsValues':
        """Create from Config object"""
        return cls(
            flash_duration_ms=config.timing.flash_duration_ms,
            isi_ms=config.timing.isi_ms,
            num_sequences=config.timing.num_sequences,
            color_scheme=config.arrows.color_scheme,
        )
    
    def apply_to_config(self, config: Config):
        """Apply these values to a Config object"""
        config.timing.flash_duration_ms = self.flash_duration_ms
        config.timing.isi_ms = self.isi_ms
        config.timing.num_sequences = self.num_sequences
        config.arrows.color_scheme = self.color_scheme


# =============================================================================
# UI Components
# =============================================================================

class Slider:
    """
    Horizontal slider for numeric values.
    
    Features:
    - Draggable handle
    - Click-to-set on track
    - Integer or float values
    - Custom formatting
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        min_value: float,
        max_value: float,
        initial_value: float,
        step: float = 1,
        label: str = "",
        format_str: str = "{:.0f}",
        font: pygame.font.Font = None,
    ):
        self.rect = rect
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.step = step
        self.label = label
        self.format_str = format_str
        self.font = font or pygame.font.Font(None, 24)
        
        # Track dimensions
        self.track_height = 6
        self.track_rect = pygame.Rect(
            rect.x + 10,
            rect.centery - self.track_height // 2,
            rect.width - 20,
            self.track_height
        )
        
        # Handle dimensions
        self.handle_width = 16
        self.handle_height = 24
        self.handle_rect = pygame.Rect(0, 0, self.handle_width, self.handle_height)
        self._update_handle_position()
        
        # State
        self.dragging = False
        self.hovered = False
        
        # Colors (duller to make arrows stand out)
        self.track_color = (30, 30, 30)
        self.track_fill_color = (40, 50, 70)
        self.handle_color = (70, 70, 70)
        self.handle_hover_color = (85, 85, 85)
        self.handle_drag_color = (50, 60, 80)
        self.label_color = (80, 80, 80)
        self.value_color = (60, 75, 95)
        
    def _update_handle_position(self):
        """Update handle position based on current value"""
        ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        x = self.track_rect.x + ratio * (self.track_rect.width - self.handle_width)
        self.handle_rect.x = int(x)
        self.handle_rect.centery = self.track_rect.centery
        
    def _value_from_x(self, x: int) -> float:
        """Calculate value from x position"""
        # Clamp x to track bounds
        x = max(self.track_rect.x, min(x, self.track_rect.right - self.handle_width))
        ratio = (x - self.track_rect.x) / (self.track_rect.width - self.handle_width)
        value = self.min_value + ratio * (self.max_value - self.min_value)
        
        # Snap to step
        value = round(value / self.step) * self.step
        return max(self.min_value, min(self.max_value, value))
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.
        
        Returns:
            True if value changed
        """
        old_value = self.value
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                if self.handle_rect.collidepoint(event.pos):
                    self.dragging = True
                elif self.track_rect.collidepoint(event.pos):
                    # Click on track - jump to position
                    self.value = self._value_from_x(event.pos[0] - self.handle_width // 2)
                    self._update_handle_position()
                    
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
                
        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.handle_rect.collidepoint(event.pos)
            
            if self.dragging:
                self.value = self._value_from_x(event.pos[0] - self.handle_width // 2)
                self._update_handle_position()
                
        return self.value != old_value
        
    def draw(self, screen: pygame.Surface):
        """Draw the slider"""
        # Draw label
        if self.label:
            label_surface = self.font.render(self.label, True, self.label_color)
            screen.blit(label_surface, (self.rect.x, self.rect.y))
            
        # Draw track background
        pygame.draw.rect(screen, self.track_color, self.track_rect, border_radius=3)
        
        # Draw filled portion
        fill_width = self.handle_rect.centerx - self.track_rect.x
        if fill_width > 0:
            fill_rect = pygame.Rect(
                self.track_rect.x,
                self.track_rect.y,
                fill_width,
                self.track_rect.height
            )
            pygame.draw.rect(screen, self.track_fill_color, fill_rect, border_radius=3)
            
        # Draw handle
        if self.dragging:
            color = self.handle_drag_color
        elif self.hovered:
            color = self.handle_hover_color
        else:
            color = self.handle_color
            
        pygame.draw.rect(screen, color, self.handle_rect, border_radius=4)
        pygame.draw.rect(screen, (45, 45, 45), self.handle_rect, 1, border_radius=4)
        
        # Draw value
        value_text = self.format_str.format(self.value)
        value_surface = self.font.render(value_text, True, self.value_color)
        value_rect = value_surface.get_rect(
            right=self.rect.right,
            centery=self.track_rect.centery
        )
        screen.blit(value_surface, value_rect)


class Dropdown:
    """
    Dropdown selector for enum values.
    
    Features:
    - Click to open/close
    - Hover highlighting
    - Supports any enum type
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        options: List[Enum],
        initial_value: Enum,
        label: str = "",
        display_names: Dict[Enum, str] = None,
        font: pygame.font.Font = None,
    ):
        self.rect = rect
        self.options = options
        self.value = initial_value
        self.label = label
        self.display_names = display_names or {opt: opt.name for opt in options}
        self.font = font or pygame.font.Font(None, 24)
        
        # Dropdown button rect
        self.button_rect = pygame.Rect(
            rect.x,
            rect.y + 25,  # Below label
            rect.width,
            30
        )
        
        # State
        self.is_open = False
        self.hovered_index = -1
        
        # Colors (duller to make arrows stand out)
        self.bg_color = (25, 25, 25)
        self.hover_color = (35, 40, 50)
        self.border_color = (45, 45, 45)
        self.text_color = (80, 80, 80)
        self.label_color = (80, 80, 80)
        self.arrow_color = (60, 60, 60)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.
        
        Returns:
            True if value changed
        """
        old_value = self.value
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.button_rect.collidepoint(event.pos):
                    self.is_open = not self.is_open
                elif self.is_open:
                    # Check if clicked on an option
                    for i, option in enumerate(self.options):
                        option_rect = pygame.Rect(
                            self.button_rect.x,
                            self.button_rect.bottom + i * 30,
                            self.button_rect.width,
                            30
                        )
                        if option_rect.collidepoint(event.pos):
                            self.value = option
                            self.is_open = False
                            break
                    else:
                        # Clicked outside - close
                        self.is_open = False
                        
        elif event.type == pygame.MOUSEMOTION:
            if self.is_open:
                self.hovered_index = -1
                for i, option in enumerate(self.options):
                    option_rect = pygame.Rect(
                        self.button_rect.x,
                        self.button_rect.bottom + i * 30,
                        self.button_rect.width,
                        30
                    )
                    if option_rect.collidepoint(event.pos):
                        self.hovered_index = i
                        break
                        
        return self.value != old_value
        
    def draw(self, screen: pygame.Surface):
        """Draw the dropdown"""
        # Draw label
        if self.label:
            label_surface = self.font.render(self.label, True, self.label_color)
            screen.blit(label_surface, (self.rect.x, self.rect.y))
            
        # Draw button
        pygame.draw.rect(screen, self.bg_color, self.button_rect)
        pygame.draw.rect(screen, self.border_color, self.button_rect, 1)
        
        # Draw current value text
        text = self.display_names.get(self.value, str(self.value))
        text_surface = self.font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect(
            left=self.button_rect.x + 10,
            centery=self.button_rect.centery
        )
        screen.blit(text_surface, text_rect)
        
        # Draw arrow
        arrow_x = self.button_rect.right - 20
        arrow_y = self.button_rect.centery
        if self.is_open:
            # Up arrow
            points = [
                (arrow_x, arrow_y + 4),
                (arrow_x + 8, arrow_y + 4),
                (arrow_x + 4, arrow_y - 4),
            ]
        else:
            # Down arrow
            points = [
                (arrow_x, arrow_y - 4),
                (arrow_x + 8, arrow_y - 4),
                (arrow_x + 4, arrow_y + 4),
            ]
        pygame.draw.polygon(screen, self.arrow_color, points)
        
        # Draw options if open
        if self.is_open:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(
                    self.button_rect.x,
                    self.button_rect.bottom + i * 30,
                    self.button_rect.width,
                    30
                )
                
                # Background
                if i == self.hovered_index:
                    pygame.draw.rect(screen, self.hover_color, option_rect)
                else:
                    pygame.draw.rect(screen, self.bg_color, option_rect)
                pygame.draw.rect(screen, self.border_color, option_rect, 1)
                
                # Text
                text = self.display_names.get(option, str(option))
                text_surface = self.font.render(text, True, self.text_color)
                text_rect = text_surface.get_rect(
                    left=option_rect.x + 10,
                    centery=option_rect.centery
                )
                screen.blit(text_surface, text_rect)
                
    def get_total_height(self) -> int:
        """Get total height when open"""
        if self.is_open:
            return self.button_rect.height + len(self.options) * 30 + 25
        return self.button_rect.height + 25


class Button:
    """Simple button with hover effect"""
    
    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font = None,
        color: Tuple[int, int, int] = (35, 35, 35),
        hover_color: Tuple[int, int, int] = (45, 50, 60),
        text_color: Tuple[int, int, int] = (85, 85, 85),
    ):
        self.rect = rect
        self.text = text
        self.font = font or pygame.font.Font(None, 28)
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.
        
        Returns:
            True if button was clicked
        """
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                return True
                
        return False
        
    def draw(self, screen: pygame.Surface):
        """Draw the button"""
        color = self.hover_color if self.hovered else self.color
        
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (55, 55, 55), self.rect, 2, border_radius=5)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)


# =============================================================================
# Settings Panel
# =============================================================================

class SettingsPanel:
    """
    Settings panel for P300 BCI parameters.
    
    Displays as a modal overlay with:
    - Flash duration slider (50-250ms)
    - ISI slider (50-300ms)
    - Number of sequences slider (1-20)
    - Color scheme dropdown
    - Real-time SOA display
    - Apply/Cancel buttons
    
    Usage:
        panel = SettingsPanel(screen_width, screen_height)
        panel.set_values(SettingsValues.from_config(config))
        panel.set_callbacks(on_apply=apply_func, on_cancel=cancel_func)
        panel.show()
        
        # In game loop:
        if panel.is_visible:
            panel.handle_event(event)
            panel.draw(screen)
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Panel dimensions
        self.panel_width = 450
        self.panel_height = 420
        self.panel_rect = pygame.Rect(
            (screen_width - self.panel_width) // 2,
            (screen_height - self.panel_height) // 2,
            self.panel_width,
            self.panel_height
        )
        
        # State
        self._is_visible = False
        self._values = SettingsValues()
        self._original_values = SettingsValues()
        
        # Callbacks
        self._on_apply: Optional[Callable[[SettingsValues], None]] = None
        self._on_cancel: Optional[Callable[[], None]] = None
        
        # Fonts
        self.title_font = pygame.font.Font(None, 36)
        self.label_font = pygame.font.Font(None, 24)
        self.info_font = pygame.font.Font(None, 22)
        
        # Colors (duller to make arrows stand out)
        self.bg_color = (20, 20, 22)
        self.border_color = (40, 40, 45)
        self.title_color = (90, 90, 95)
        self.info_color = (60, 70, 85)
        
        # Create UI elements
        self._create_ui_elements()
        
    def _create_ui_elements(self):
        """Create all UI elements"""
        x = self.panel_rect.x + 30
        y = self.panel_rect.y + 60
        width = self.panel_width - 60
        
        # Flash duration slider
        self.flash_slider = Slider(
            rect=pygame.Rect(x, y, width, 50),
            min_value=50,
            max_value=250,
            initial_value=self._values.flash_duration_ms,
            step=10,
            label="Flash Duration",
            format_str="{:.0f}ms",
            font=self.label_font,
        )
        y += 60
        
        # ISI slider
        self.isi_slider = Slider(
            rect=pygame.Rect(x, y, width, 50),
            min_value=50,
            max_value=300,
            initial_value=self._values.isi_ms,
            step=10,
            label="Inter-Stimulus Interval (ISI)",
            format_str="{:.0f}ms",
            font=self.label_font,
        )
        y += 60
        
        # Sequences slider
        self.sequences_slider = Slider(
            rect=pygame.Rect(x, y, width, 50),
            min_value=1,
            max_value=20,
            initial_value=self._values.num_sequences,
            step=1,
            label="Sequences per Selection",
            format_str="{:.0f}",
            font=self.label_font,
        )
        y += 70
        
        # Color scheme dropdown
        color_scheme_names = {
            ColorScheme.GRAY_WHITE: "Gray → White (Default)",
            ColorScheme.GREEN_BLUE: "Blue → Green (Comfort)",
            ColorScheme.INVERTED: "Light → Dark (Inverted)",
        }
        self.color_dropdown = Dropdown(
            rect=pygame.Rect(x, y, width, 60),
            options=list(ColorScheme),
            initial_value=self._values.color_scheme,
            label="Color Scheme",
            display_names=color_scheme_names,
            font=self.label_font,
        )
        y += 80
        
        # Buttons
        button_width = 100
        button_height = 36
        button_y = self.panel_rect.bottom - 60
        
        self.apply_button = Button(
            rect=pygame.Rect(
                self.panel_rect.centerx - button_width - 20,
                button_y,
                button_width,
                button_height
            ),
            text="Apply",
            font=self.label_font,
            color=(25, 45, 25),
            hover_color=(35, 60, 35),
        )
        
        self.cancel_button = Button(
            rect=pygame.Rect(
                self.panel_rect.centerx + 20,
                button_y,
                button_width,
                button_height
            ),
            text="Cancel",
            font=self.label_font,
            color=(45, 25, 25),
            hover_color=(60, 35, 35),
        )
        
    def set_values(self, values: SettingsValues):
        """Set current values"""
        self._values = values
        self._original_values = SettingsValues(
            flash_duration_ms=values.flash_duration_ms,
            isi_ms=values.isi_ms,
            num_sequences=values.num_sequences,
            color_scheme=values.color_scheme,
        )
        self._update_ui_from_values()
        
    def _update_ui_from_values(self):
        """Update UI elements to reflect current values"""
        self.flash_slider.value = self._values.flash_duration_ms
        self.flash_slider._update_handle_position()
        
        self.isi_slider.value = self._values.isi_ms
        self.isi_slider._update_handle_position()
        
        self.sequences_slider.value = self._values.num_sequences
        self.sequences_slider._update_handle_position()
        
        self.color_dropdown.value = self._values.color_scheme
        
    def _update_values_from_ui(self):
        """Update values from UI elements"""
        self._values.flash_duration_ms = int(self.flash_slider.value)
        self._values.isi_ms = int(self.isi_slider.value)
        self._values.num_sequences = int(self.sequences_slider.value)
        self._values.color_scheme = self.color_dropdown.value
        
    def set_callbacks(
        self,
        on_apply: Optional[Callable[[SettingsValues], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ):
        """Set callback functions"""
        self._on_apply = on_apply
        self._on_cancel = on_cancel
        
    def show(self):
        """Show the settings panel"""
        self._is_visible = True
        
    def hide(self):
        """Hide the settings panel"""
        self._is_visible = False
        self.color_dropdown.is_open = False
        
    @property
    def is_visible(self) -> bool:
        """Whether the panel is currently visible"""
        return self._is_visible
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.
        
        Returns:
            True if event was consumed (panel is visible)
        """
        if not self._is_visible:
            return False
            
        # Keyboard shortcuts
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self._apply()
                return True
            elif event.key == pygame.K_ESCAPE:
                self._cancel()
                return True
                
        # UI elements
        self.flash_slider.handle_event(event)
        self.isi_slider.handle_event(event)
        self.sequences_slider.handle_event(event)
        self.color_dropdown.handle_event(event)
        
        # Update values after slider changes
        self._update_values_from_ui()
        
        # Buttons
        if self.apply_button.handle_event(event):
            self._apply()
        if self.cancel_button.handle_event(event):
            self._cancel()
            
        return True
        
    def _apply(self):
        """Apply settings and close"""
        self._update_values_from_ui()
        self.hide()
        if self._on_apply:
            self._on_apply(self._values)
            
    def _cancel(self):
        """Cancel and restore original values"""
        self._values = self._original_values
        self._update_ui_from_values()
        self.hide()
        if self._on_cancel:
            self._on_cancel()
            
    def draw(self, screen: pygame.Surface):
        """Draw the settings panel"""
        if not self._is_visible:
            return
            
        # Draw dimmed background
        dim_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        dim_surface.fill((0, 0, 0, 180))
        screen.blit(dim_surface, (0, 0))
        
        # Draw panel background
        pygame.draw.rect(screen, self.bg_color, self.panel_rect, border_radius=10)
        pygame.draw.rect(screen, self.border_color, self.panel_rect, 2, border_radius=10)
        
        # Draw title
        title_surface = self.title_font.render("Settings", True, self.title_color)
        title_rect = title_surface.get_rect(
            centerx=self.panel_rect.centerx,
            top=self.panel_rect.top + 15
        )
        screen.blit(title_surface, title_rect)
        
        # Draw UI elements
        self.flash_slider.draw(screen)
        self.isi_slider.draw(screen)
        self.sequences_slider.draw(screen)
        self.color_dropdown.draw(screen)
        
        # Draw calculated info
        self._draw_info(screen)
        
        # Draw buttons
        self.apply_button.draw(screen)
        self.cancel_button.draw(screen)
        
    def _draw_info(self, screen: pygame.Surface):
        """Draw calculated timing info"""
        # SOA and flash rate
        soa = self._values.soa_ms
        rate = self._values.flash_rate_hz
        
        info_y = self.panel_rect.bottom - 100
        
        info_text = f"SOA: {soa}ms  |  Flash Rate: {rate:.1f}Hz per arrow"
        info_surface = self.info_font.render(info_text, True, self.info_color)
        info_rect = info_surface.get_rect(
            centerx=self.panel_rect.centerx,
            top=info_y
        )
        screen.blit(info_surface, info_rect)
        
        # Selection duration estimate
        # 4 arrows × sequences × SOA + (sequences-1) × pause
        num_seq = self._values.num_sequences
        pause_ms = 200  # inter_sequence_pause_ms
        duration_ms = 4 * num_seq * soa + (num_seq - 1) * pause_ms
        duration_s = duration_ms / 1000
        
        duration_text = f"Est. selection time: {duration_s:.1f}s"
        duration_surface = self.info_font.render(duration_text, True, (50, 65, 50))
        duration_rect = duration_surface.get_rect(
            centerx=self.panel_rect.centerx,
            top=info_y + 22
        )
        screen.blit(duration_surface, duration_rect)


# =============================================================================
# Testing / Demo
# =============================================================================

def demo():
    """Run a demo of the settings panel"""
    pygame.init()
    
    screen_width, screen_height = 1024, 768
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Settings Panel Demo")
    
    clock = pygame.time.Clock()
    
    # Create settings panel
    panel = SettingsPanel(screen_width, screen_height)
    
    # Set initial values
    initial_values = SettingsValues(
        flash_duration_ms=100,
        isi_ms=125,
        num_sequences=10,
        color_scheme=ColorScheme.GRAY_WHITE,
    )
    panel.set_values(initial_values)
    
    # Set callbacks
    def on_apply(values: SettingsValues):
        print("Settings applied:")
        print(f"  Flash: {values.flash_duration_ms}ms")
        print(f"  ISI: {values.isi_ms}ms")
        print(f"  Sequences: {values.num_sequences}")
        print(f"  Color: {values.color_scheme.name}")
        print(f"  SOA: {values.soa_ms}ms")
        
    def on_cancel():
        print("Settings cancelled")
        
    panel.set_callbacks(on_apply=on_apply, on_cancel=on_cancel)
    
    # Show panel
    panel.show()
    
    running = True
    font = pygame.font.Font(None, 32)
    
    print("Settings Panel Demo")
    print("  S - Toggle settings panel")
    print("  Enter - Apply settings")
    print("  Escape - Cancel")
    print()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s and not panel.is_visible:
                    panel.show()
                    
            # Let panel handle events first
            if not panel.handle_event(event):
                # Panel didn't consume event
                pass
                
        # Draw
        screen.fill((10, 10, 12))
        
        # Draw background info
        if not panel.is_visible:
            text = font.render("Press S to open settings", True, (50, 50, 50))
            rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(text, rect)
            
        # Draw panel
        panel.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()


if __name__ == "__main__":
    demo()
