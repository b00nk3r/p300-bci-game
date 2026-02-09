"""
Settings Panel - Pixel Art Style
=================================
Runtime configuration panel for P300 BCI parameters.
Redesigned with bright pixel art aesthetic to match game style.
"""

import pygame
from typing import Callable, Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from config import Config, ColorScheme, FlashPattern


@dataclass
class SettingsValues:
    """Current settings values that can be modified"""
    flash_duration_ms: int = 100
    isi_ms: int = 125
    num_sequences: int = 10
    color_scheme: ColorScheme = ColorScheme.GRAY_WHITE
    dullness: int = 5
    
    @property
    def soa_ms(self) -> int:
        return self.flash_duration_ms + self.isi_ms
    
    @property
    def flash_rate_hz(self) -> float:
        return 1000 / self.soa_ms if self.soa_ms > 0 else 0
    
    @classmethod
    def from_config(cls, config: Config) -> 'SettingsValues':
        return cls(
            flash_duration_ms=config.timing.flash_duration_ms,
            isi_ms=config.timing.isi_ms,
            num_sequences=config.timing.num_sequences,
            color_scheme=config.arrows.color_scheme,
            dullness=config.game.dullness,
        )
    
    def apply_to_config(self, config: Config):
        config.timing.flash_duration_ms = self.flash_duration_ms
        config.timing.isi_ms = self.isi_ms
        config.timing.num_sequences = self.num_sequences
        config.arrows.color_scheme = self.color_scheme
        config.game.dullness = self.dullness


class PixelColors:
    """Bright pixel art color palette - lighter and more vibrant"""
    # Panel backgrounds - brighter
    PANEL_BG = (85, 90, 100)
    PANEL_BORDER_LIGHT = (160, 165, 175)
    PANEL_BORDER_DARK = (45, 48, 55)
    PANEL_INNER = (95, 100, 110)
    PANEL_HIGHLIGHT = (180, 185, 195)
    
    # Text - bright and readable
    TITLE = (255, 255, 255)
    TITLE_SHADOW = (40, 45, 55)
    LABEL = (220, 225, 235)
    VALUE = (180, 220, 255)
    INFO = (170, 200, 230)
    
    # Sliders - more vibrant
    TRACK_BG = (55, 60, 70)
    TRACK_FILL = (120, 170, 220)
    TRACK_FILL_GLOW = (150, 200, 250)
    TRACK_BORDER = (100, 105, 115)
    HANDLE = (180, 185, 195)
    HANDLE_LIGHT = (220, 225, 235)
    HANDLE_DARK = (100, 105, 115)
    HANDLE_HOVER = (200, 205, 215)
    HANDLE_ACTIVE = (140, 190, 240)
    
    # Buttons - brighter
    BUTTON_BG = (100, 105, 120)
    BUTTON_BORDER_LIGHT = (160, 165, 180)
    BUTTON_BORDER_DARK = (55, 60, 70)
    BUTTON_HOVER = (120, 125, 140)
    BUTTON_TEXT = (255, 255, 255)
    
    # Apply button - green accent
    APPLY_BG = (80, 140, 100)
    APPLY_LIGHT = (130, 200, 150)
    APPLY_DARK = (45, 80, 55)
    APPLY_HOVER = (100, 170, 120)
    
    # Cancel button - red accent
    CANCEL_BG = (140, 80, 90)
    CANCEL_LIGHT = (200, 130, 140)
    CANCEL_DARK = (80, 45, 50)
    CANCEL_HOVER = (170, 100, 110)
    
    # Dropdown - brighter
    DROPDOWN_BG = (90, 95, 105)
    DROPDOWN_HOVER = (110, 130, 160)
    DROPDOWN_BORDER = (130, 135, 145)
    
    # Decorative
    CORNER_ACCENT = (200, 210, 220)
    DIVIDER = (140, 145, 155)


def draw_pixel_rect(surface, rect, fill_color, light_color, dark_color, border_width=4):
    """Draw a raised pixel art style rectangle"""
    pygame.draw.rect(surface, fill_color, rect)
    # Top and left edges (light)
    pygame.draw.rect(surface, light_color, (rect.x, rect.y, rect.width, border_width))
    pygame.draw.rect(surface, light_color, (rect.x, rect.y, border_width, rect.height))
    # Bottom and right edges (dark)
    pygame.draw.rect(surface, dark_color, (rect.x, rect.bottom - border_width, rect.width, border_width))
    pygame.draw.rect(surface, dark_color, (rect.right - border_width, rect.y, border_width, rect.height))


def draw_pixel_rect_inset(surface, rect, fill_color, light_color, dark_color, border_width=3):
    """Draw an inset pixel art style rectangle"""
    pygame.draw.rect(surface, fill_color, rect)
    # Top and left edges (dark - inset)
    pygame.draw.rect(surface, dark_color, (rect.x, rect.y, rect.width, border_width))
    pygame.draw.rect(surface, dark_color, (rect.x, rect.y, border_width, rect.height))
    # Bottom and right edges (light - inset)
    pygame.draw.rect(surface, light_color, (rect.x, rect.bottom - border_width, rect.width, border_width))
    pygame.draw.rect(surface, light_color, (rect.right - border_width, rect.y, border_width, rect.height))


def draw_pixel_border_double(surface, rect, outer_color, inner_color, width=6):
    """Draw a double pixel border for extra chunky look"""
    # Outer border
    pygame.draw.rect(surface, outer_color, rect, width)
    # Inner border
    inner_rect = rect.inflate(-width * 2, -width * 2)
    pygame.draw.rect(surface, inner_color, inner_rect, width // 2)


def draw_corner_decorations(surface, rect, color, size=16):
    """Draw pixel art corner decorations"""
    corners = [
        (rect.x, rect.y),  # Top-left
        (rect.right - size, rect.y),  # Top-right
        (rect.x, rect.bottom - size),  # Bottom-left
        (rect.right - size, rect.bottom - size),  # Bottom-right
    ]
    for cx, cy in corners:
        # L-shaped corner accent
        pygame.draw.rect(surface, color, (cx, cy, size, 4))
        pygame.draw.rect(surface, color, (cx, cy, 4, size))


class PixelSlider:
    def __init__(self, rect, min_value, max_value, initial_value, step=1, label="", format_str="{:.0f}", font=None):
        self.rect = rect
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.step = step
        self.label = label
        self.format_str = format_str
        self.font = font or pygame.font.Font(None, 36)
        
        # Larger track for pixel art look
        self.track_height = 24
        self.track_rect = pygame.Rect(rect.x, rect.centery + 12, rect.width - 140, self.track_height)
        self.handle_width = 36
        self.handle_height = 48
        self.handle_rect = pygame.Rect(0, 0, self.handle_width, self.handle_height)
        self._update_handle_position()
        self.dragging = False
        self.hovered = False
        
    def _update_handle_position(self):
        if self.max_value == self.min_value:
            ratio = 0
        else:
            ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        x = self.track_rect.x + ratio * (self.track_rect.width - self.handle_width)
        self.handle_rect.x = int(x)
        self.handle_rect.centery = self.track_rect.centery
        
    def _value_from_x(self, x):
        x = max(self.track_rect.x, min(x, self.track_rect.right - self.handle_width))
        if self.track_rect.width == self.handle_width:
            ratio = 0
        else:
            ratio = (x - self.track_rect.x) / (self.track_rect.width - self.handle_width)
        value = self.min_value + ratio * (self.max_value - self.min_value)
        value = round(value / self.step) * self.step
        return max(self.min_value, min(self.max_value, value))
        
    def handle_event(self, event):
        old_value = self.value
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
            elif self.track_rect.collidepoint(event.pos):
                self.value = self._value_from_x(event.pos[0] - self.handle_width // 2)
                self._update_handle_position()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.handle_rect.collidepoint(event.pos)
            if self.dragging:
                self.value = self._value_from_x(event.pos[0] - self.handle_width // 2)
                self._update_handle_position()
        return self.value != old_value
        
    def draw(self, screen):
        # Label
        if self.label:
            label_surface = self.font.render(self.label, True, PixelColors.LABEL)
            screen.blit(label_surface, (self.rect.x, self.rect.y))
        
        # Track background with chunky border
        draw_pixel_rect_inset(screen, self.track_rect, PixelColors.TRACK_BG, 
                             PixelColors.TRACK_BORDER, PixelColors.PANEL_BORDER_DARK, 4)
        
        # Filled portion with glow effect
        fill_width = self.handle_rect.centerx - self.track_rect.x - 4
        if fill_width > 0:
            fill_rect = pygame.Rect(self.track_rect.x + 4, self.track_rect.y + 4, 
                                   fill_width, self.track_rect.height - 8)
            pygame.draw.rect(screen, PixelColors.TRACK_FILL, fill_rect)
            # Top highlight
            pygame.draw.rect(screen, PixelColors.TRACK_FILL_GLOW, 
                           (fill_rect.x, fill_rect.y, fill_rect.width, 4))
        
        # Handle with chunky pixel art style
        if self.dragging:
            handle_color = PixelColors.HANDLE_ACTIVE
        elif self.hovered:
            handle_color = PixelColors.HANDLE_HOVER
        else:
            handle_color = PixelColors.HANDLE
        
        draw_pixel_rect(screen, self.handle_rect, handle_color, 
                       PixelColors.HANDLE_LIGHT, PixelColors.HANDLE_DARK, 4)
        
        # Handle grip lines (pixel art detail)
        grip_x = self.handle_rect.centerx
        grip_y1 = self.handle_rect.y + 12
        grip_y2 = self.handle_rect.bottom - 12
        for offset in [-6, 0, 6]:
            pygame.draw.rect(screen, PixelColors.HANDLE_DARK, 
                           (grip_x + offset - 2, grip_y1, 4, grip_y2 - grip_y1))
        
        # Value display with background
        value_text = self.format_str.format(self.value)
        value_surface = self.font.render(value_text, True, PixelColors.VALUE)
        value_bg = pygame.Rect(self.rect.right - 120, self.track_rect.y - 4, 
                              110, self.track_rect.height + 8)
        draw_pixel_rect_inset(screen, value_bg, PixelColors.TRACK_BG, 
                             PixelColors.TRACK_BORDER, PixelColors.PANEL_BORDER_DARK, 3)
        value_rect = value_surface.get_rect(center=value_bg.center)
        screen.blit(value_surface, value_rect)


class PixelDropdown:
    def __init__(self, rect, options, initial_value, label="", display_names=None, font=None):
        self.rect = rect
        self.options = options
        self.value = initial_value
        self.label = label
        self.display_names = display_names or {opt: opt.name for opt in options}
        self.font = font or pygame.font.Font(None, 36)
        
        # Larger button for pixel art look
        self.button_height = 56
        self.button_rect = pygame.Rect(rect.x, rect.y + 40, rect.width, self.button_height)
        self.is_open = False
        self.hovered_index = -1
        
    def handle_event(self, event):
        old_value = self.value
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.button_rect.collidepoint(event.pos):
                self.is_open = not self.is_open
            elif self.is_open:
                for i, option in enumerate(self.options):
                    option_rect = pygame.Rect(self.button_rect.x, 
                                             self.button_rect.bottom + i * self.button_height,
                                             self.button_rect.width, self.button_height)
                    if option_rect.collidepoint(event.pos):
                        self.value = option
                        self.is_open = False
                        break
                else:
                    self.is_open = False
        elif event.type == pygame.MOUSEMOTION and self.is_open:
            self.hovered_index = -1
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.button_rect.x, 
                                         self.button_rect.bottom + i * self.button_height,
                                         self.button_rect.width, self.button_height)
                if option_rect.collidepoint(event.pos):
                    self.hovered_index = i
                    break
        return self.value != old_value
        
    def draw(self, screen):
        # Label
        if self.label:
            label_surface = self.font.render(self.label, True, PixelColors.LABEL)
            screen.blit(label_surface, (self.rect.x, self.rect.y))
        
        # Main button with chunky border
        draw_pixel_rect(screen, self.button_rect, PixelColors.DROPDOWN_BG,
                       PixelColors.PANEL_BORDER_LIGHT, PixelColors.PANEL_BORDER_DARK, 4)
        
        # Current value text
        text = self.display_names.get(self.value, str(self.value))
        text_surface = self.font.render(text, True, PixelColors.BUTTON_TEXT)
        text_rect = text_surface.get_rect(left=self.button_rect.x + 20, 
                                         centery=self.button_rect.centery)
        screen.blit(text_surface, text_rect)
        
        # Pixel art arrow
        arrow_x = self.button_rect.right - 45
        arrow_y = self.button_rect.centery
        arrow_size = 12
        
        if self.is_open:
            # Up arrow (pixel art triangles)
            for i in range(arrow_size // 3):
                w = (arrow_size - i * 3) * 2
                pygame.draw.rect(screen, PixelColors.LABEL, 
                               (arrow_x - w // 2 + arrow_size, arrow_y + i * 3 - 6, w, 3))
        else:
            # Down arrow
            for i in range(arrow_size // 3):
                w = (arrow_size - i * 3) * 2
                pygame.draw.rect(screen, PixelColors.LABEL, 
                               (arrow_x - w // 2 + arrow_size, arrow_y - i * 3 + 3, w, 3))
        
        # Dropdown options
        if self.is_open:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.button_rect.x, 
                                         self.button_rect.bottom + i * self.button_height,
                                         self.button_rect.width, self.button_height)
                bg_color = PixelColors.DROPDOWN_HOVER if i == self.hovered_index else PixelColors.DROPDOWN_BG
                draw_pixel_rect(screen, option_rect, bg_color, 
                              PixelColors.DROPDOWN_BORDER, PixelColors.PANEL_BORDER_DARK, 3)
                
                text = self.display_names.get(option, str(option))
                text_surface = self.font.render(text, True, PixelColors.BUTTON_TEXT)
                text_rect = text_surface.get_rect(left=option_rect.x + 20, 
                                                 centery=option_rect.centery)
                screen.blit(text_surface, text_rect)


class PixelButton:
    def __init__(self, rect, text, font=None, 
                 bg_color=None, light_color=None, dark_color=None, 
                 hover_color=None, text_color=None):
        self.rect = rect
        self.text = text
        self.font = font or pygame.font.Font(None, 40)
        self.bg_color = bg_color or PixelColors.BUTTON_BG
        self.light_color = light_color or PixelColors.BUTTON_BORDER_LIGHT
        self.dark_color = dark_color or PixelColors.BUTTON_BORDER_DARK
        self.hover_color = hover_color or PixelColors.BUTTON_HOVER
        self.text_color = text_color or PixelColors.BUTTON_TEXT
        self.hovered = False
        self.pressed = False
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed:
                self.pressed = False
                if self.rect.collidepoint(event.pos):
                    return True
        return False
        
    def draw(self, screen):
        if self.pressed:
            draw_pixel_rect_inset(screen, self.rect, self.bg_color, 
                                 self.light_color, self.dark_color, 5)
        else:
            bg = self.hover_color if self.hovered else self.bg_color
            draw_pixel_rect(screen, self.rect, bg, self.light_color, self.dark_color, 5)
        
        # Text with shadow for pixel art effect
        shadow_surface = self.font.render(self.text, True, PixelColors.TITLE_SHADOW)
        shadow_rect = shadow_surface.get_rect(center=(self.rect.centerx + 2, self.rect.centery + 2))
        if self.pressed:
            shadow_rect.x += 2
            shadow_rect.y += 2
        screen.blit(shadow_surface, shadow_rect)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        if self.pressed:
            text_rect.x += 2
            text_rect.y += 2
        screen.blit(text_surface, text_rect)


class SettingsPanel:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # BIGGER panel size
        self.panel_width = 950
        self.panel_height = 950
        self.panel_rect = pygame.Rect(
            (screen_width - self.panel_width) // 2, 
            (screen_height - self.panel_height) // 2, 
            self.panel_width, 
            self.panel_height
        )
        
        # Inner panel for content
        self.inner_margin = 20
        self.inner_rect = pygame.Rect(
            self.panel_rect.x + self.inner_margin, 
            self.panel_rect.y + self.inner_margin, 
            self.panel_width - self.inner_margin * 2, 
            self.panel_height - self.inner_margin * 2
        )
        
        self._is_visible = False
        self._values = SettingsValues()
        self._original_values = SettingsValues()
        self._on_apply = None
        self._on_cancel = None
        
        # Larger fonts for pixel art style
        self.title_font = pygame.font.Font(None, 72)
        self.label_font = pygame.font.Font(None, 38)
        self.info_font = pygame.font.Font(None, 34)
        
        self._create_ui_elements()
        
    def _create_ui_elements(self):
        x = self.inner_rect.x + 50
        y = self.inner_rect.y + 130
        width = self.inner_rect.width - 100
        
        # Larger spacing between elements
        self.flash_slider = PixelSlider(
            pygame.Rect(x, y, width, 90), 
            50, 250, self._values.flash_duration_ms, 10, 
            "Flash Duration", "{:.0f} ms", self.label_font
        )
        y += 110
        
        self.isi_slider = PixelSlider(
            pygame.Rect(x, y, width, 90), 
            50, 300, self._values.isi_ms, 10, 
            "Inter-Stimulus Interval (ISI)", "{:.0f} ms", self.label_font
        )
        y += 110
        
        self.sequences_slider = PixelSlider(
            pygame.Rect(x, y, width, 90), 
            1, 20, self._values.num_sequences, 1, 
            "Sequences per Selection", "{:.0f}", self.label_font
        )
        y += 110
        
        self.dullness_slider = PixelSlider(
            pygame.Rect(x, y, width, 90), 
            1, 5, self._values.dullness, 1, 
            "Game Dullness (Arrows Stand Out)", "{:.0f}", self.label_font
        )
        y += 130
        
        color_scheme_names = {
            ColorScheme.GRAY_WHITE: "Gray -> White (Default)", 
            ColorScheme.GREEN_BLUE: "Blue -> Green (Comfort)", 
            ColorScheme.INVERTED: "Light -> Dark (Inverted)"
        }
        self.color_dropdown = PixelDropdown(
            pygame.Rect(x, y, width, 110), 
            list(ColorScheme), self._values.color_scheme, 
            "Color Scheme", color_scheme_names, self.label_font
        )
        
        # Larger buttons
        button_width = 200
        button_height = 70
        button_y = self.panel_rect.bottom - 120
        
        self.apply_button = PixelButton(
            pygame.Rect(self.panel_rect.centerx - button_width - 40, button_y, 
                       button_width, button_height), 
            "APPLY", self.label_font, 
            PixelColors.APPLY_BG, PixelColors.APPLY_LIGHT, 
            PixelColors.APPLY_DARK, PixelColors.APPLY_HOVER
        )
        
        self.cancel_button = PixelButton(
            pygame.Rect(self.panel_rect.centerx + 40, button_y, 
                       button_width, button_height), 
            "CANCEL", self.label_font, 
            PixelColors.CANCEL_BG, PixelColors.CANCEL_LIGHT, 
            PixelColors.CANCEL_DARK, PixelColors.CANCEL_HOVER
        )
        
    def set_values(self, values):
        self._values = values
        self._original_values = SettingsValues(
            flash_duration_ms=values.flash_duration_ms, 
            isi_ms=values.isi_ms, 
            num_sequences=values.num_sequences, 
            color_scheme=values.color_scheme,
            dullness=values.dullness
        )
        self._update_ui_from_values()
        
    def _update_ui_from_values(self):
        self.flash_slider.value = self._values.flash_duration_ms
        self.flash_slider._update_handle_position()
        self.isi_slider.value = self._values.isi_ms
        self.isi_slider._update_handle_position()
        self.sequences_slider.value = self._values.num_sequences
        self.sequences_slider._update_handle_position()
        self.dullness_slider.value = self._values.dullness
        self.dullness_slider._update_handle_position()
        self.color_dropdown.value = self._values.color_scheme
        
    def _update_values_from_ui(self):
        self._values.flash_duration_ms = int(self.flash_slider.value)
        self._values.isi_ms = int(self.isi_slider.value)
        self._values.num_sequences = int(self.sequences_slider.value)
        self._values.dullness = int(self.dullness_slider.value)
        self._values.color_scheme = self.color_dropdown.value
        
    def set_callbacks(self, on_apply=None, on_cancel=None):
        self._on_apply = on_apply
        self._on_cancel = on_cancel
        
    def show(self):
        self._is_visible = True
        
    def hide(self):
        self._is_visible = False
        self.color_dropdown.is_open = False
        
    @property
    def is_visible(self):
        return self._is_visible
        
    def handle_event(self, event):
        if not self._is_visible:
            return False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self._apply()
                return True
            elif event.key == pygame.K_ESCAPE:
                self._cancel()
                return True
                
        self.flash_slider.handle_event(event)
        self.isi_slider.handle_event(event)
        self.sequences_slider.handle_event(event)
        self.dullness_slider.handle_event(event)
        self.color_dropdown.handle_event(event)
        self._update_values_from_ui()
        
        if self.apply_button.handle_event(event):
            self._apply()
        if self.cancel_button.handle_event(event):
            self._cancel()
            
        return True
        
    def _apply(self):
        self._update_values_from_ui()
        self.hide()
        if self._on_apply:
            self._on_apply(self._values)
            
    def _cancel(self):
        self._values = self._original_values
        self._update_ui_from_values()
        self.hide()
        if self._on_cancel:
            self._on_cancel()
            
    def _draw_pixel_title(self, screen):
        """Draw title with pixel art styling"""
        title = "SETTINGS"
        title_y = self.inner_rect.y + 30
        
        # Shadow for depth
        shadow_surface = self.title_font.render(title, True, PixelColors.TITLE_SHADOW)
        shadow_rect = shadow_surface.get_rect(centerx=self.panel_rect.centerx + 4, top=title_y + 4)
        screen.blit(shadow_surface, shadow_rect)
        
        # Main title
        title_surface = self.title_font.render(title, True, PixelColors.TITLE)
        title_rect = title_surface.get_rect(centerx=self.panel_rect.centerx, top=title_y)
        screen.blit(title_surface, title_rect)
        
        # Decorative pixel line under title
        line_y = title_rect.bottom + 15
        line_width = 350
        line_x = self.panel_rect.centerx - line_width // 2
        
        # Multi-layer line for pixel art effect
        pygame.draw.rect(screen, PixelColors.PANEL_HIGHLIGHT, 
                        (line_x, line_y, line_width, 6))
        pygame.draw.rect(screen, PixelColors.PANEL_BORDER_LIGHT, 
                        (line_x + 10, line_y + 6, line_width - 20, 3))
        pygame.draw.rect(screen, PixelColors.PANEL_BORDER_DARK, 
                        (line_x + 20, line_y + 9, line_width - 40, 2))
            
    def draw(self, screen):
        if not self._is_visible:
            return
            
        # Dim background
        dim_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        dim_surface.fill((0, 0, 0, 180))
        screen.blit(dim_surface, (0, 0))
        
        # Outer border frame (chunky pixel art style)
        outer_border = 12
        outer_rect = pygame.Rect(
            self.panel_rect.x - outer_border, 
            self.panel_rect.y - outer_border, 
            self.panel_rect.width + outer_border * 2, 
            self.panel_rect.height + outer_border * 2
        )
        
        # Multiple border layers for chunky pixel effect
        draw_pixel_rect(screen, outer_rect, PixelColors.PANEL_BORDER_DARK, 
                       PixelColors.PANEL_HIGHLIGHT, (25, 28, 35), outer_border)
        
        # Main panel
        draw_pixel_rect(screen, self.panel_rect, PixelColors.PANEL_BG, 
                       PixelColors.PANEL_BORDER_LIGHT, PixelColors.PANEL_BORDER_DARK, 8)
        
        # Inner panel
        draw_pixel_rect_inset(screen, self.inner_rect, PixelColors.PANEL_INNER, 
                             PixelColors.PANEL_BORDER_LIGHT, PixelColors.PANEL_BORDER_DARK, 5)
        
        # Corner decorations
        draw_corner_decorations(screen, self.panel_rect, PixelColors.CORNER_ACCENT, 20)
        
        # Title
        self._draw_pixel_title(screen)
        
        # UI Elements
        self.flash_slider.draw(screen)
        self.isi_slider.draw(screen)
        self.sequences_slider.draw(screen)
        self.dullness_slider.draw(screen)
        self.color_dropdown.draw(screen)
        
        # Info box
        self._draw_info(screen)
        
        # Buttons
        self.apply_button.draw(screen)
        self.cancel_button.draw(screen)
        
    def _draw_info(self, screen):
        """Draw the calculated timing info box"""
        soa = self._values.soa_ms
        rate = self._values.flash_rate_hz
        
        info_y = self.panel_rect.bottom - 210
        info_box = pygame.Rect(
            self.inner_rect.x + 50, 
            info_y - 15, 
            self.inner_rect.width - 100, 
            90
        )
        
        # Info box with pixel art style
        draw_pixel_rect_inset(screen, info_box, PixelColors.TRACK_BG, 
                             PixelColors.TRACK_BORDER, PixelColors.PANEL_BORDER_DARK, 4)
        
        # Timing info
        info_text = f"SOA: {soa} ms   |   Flash Rate: {rate:.1f} Hz per arrow"
        info_surface = self.info_font.render(info_text, True, PixelColors.INFO)
        info_rect = info_surface.get_rect(centerx=self.panel_rect.centerx, top=info_y)
        screen.blit(info_surface, info_rect)
        
        # Duration estimate
        num_seq = self._values.num_sequences
        duration_ms = 4 * num_seq * soa + (num_seq - 1) * 200
        duration_s = duration_ms / 1000
        
        duration_text = f"Estimated selection time: {duration_s:.1f}s"
        duration_surface = self.info_font.render(duration_text, True, (140, 210, 160))
        duration_rect = duration_surface.get_rect(centerx=self.panel_rect.centerx, top=info_y + 40)
        screen.blit(duration_surface, duration_rect)


def demo():
    """Demo function to test the settings panel"""
    pygame.init()
    screen_width, screen_height = 1920, 1080
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pixel Art Settings Panel Demo")
    clock = pygame.time.Clock()
    
    panel = SettingsPanel(screen_width, screen_height)
    initial_values = SettingsValues(
        flash_duration_ms=100, 
        isi_ms=125, 
        num_sequences=10, 
        color_scheme=ColorScheme.GRAY_WHITE
    )
    panel.set_values(initial_values)
    
    def on_apply(values):
        print(f"Settings applied: Flash={values.flash_duration_ms}ms, "
              f"ISI={values.isi_ms}ms, Seq={values.num_sequences}")
    
    def on_cancel():
        print("Settings cancelled")
    
    panel.set_callbacks(on_apply=on_apply, on_cancel=on_cancel)
    panel.show()
    
    running = True
    font = pygame.font.Font(None, 48)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s and not panel.is_visible:
                    panel.show()
            panel.handle_event(event)
        
        screen.fill((10, 10, 12))
        
        if not panel.is_visible:
            text = font.render("Press S to open settings", True, (120, 120, 120))
            rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(text, rect)
        
        panel.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    demo()