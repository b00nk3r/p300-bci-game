#!/usr/bin/env python3
"""Generate a comprehensive PDF document describing the P300 BCI Game."""

from fpdf import FPDF

class GameDocPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "P300 BCI Game - Technical Documentation", align="C")
            self.ln(4)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), self.w - 10, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, num, title):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(30, 30, 30)
        self.ln(4)
        self.cell(0, 12, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(50, 100, 180)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(50, 50, 50)
        self.ln(2)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(70, 70, 70)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet_point(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(4, 5.5, "-")
        self.multi_cell(0, 5.5, f" {text}")
        self.ln(1)

    def code_block(self, text, width=None):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        w = width or (self.w - 20)
        for line in text.split("\n"):
            self.set_x(12)
            self.cell(w - 4, 4.5, line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def key_value(self, key, value, indent=10):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.set_x(self.get_x() + indent)
        kw = self.get_string_width(key + ": ") + 2
        self.cell(kw, 5.5, f"{key}: ")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, value)
        self.ln(0.5)

    def table_row(self, cells, widths, bold=False, fill=False):
        self.set_font("Helvetica", "B" if bold else "", 9)
        if fill:
            self.set_fill_color(230, 238, 250)
        self.set_text_color(40, 40, 40)
        h = 6
        x_start = self.get_x()
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.cell(w, h, str(cell), border=1, fill=fill)
        self.ln(h)


def build_pdf():
    pdf = GameDocPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 18, "P300 BCI Game", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Technical Documentation & Architecture Reference", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(50, 100, 180)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(14)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(60, 60, 60)
    lines = [
        "Salem State University",
        "Undergraduate Capstone Project",
        "",
        "A Brain-Computer Interface game using P300 evoked potentials",
        "to control a maze character through flashing arrow stimuli.",
        "",
        "Python  |  Pygame  |  NumPy  |  SciPy",
    ]
    for line in lines:
        pdf.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 14, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    toc = [
        ("1", "Project Overview"),
        ("2", "How the Game Looks (Visual Description)"),
        ("3", "Technology Stack & Dependencies"),
        ("4", "Project Structure & File Layout"),
        ("5", "Architecture & Data Flow"),
        ("6", "Configuration System"),
        ("7", "Stimulus Module (src/stimulus/)"),
        ("8", "Game Module (src/game/)"),
        ("9", "UI Module (src/ui/)"),
        ("10", "Data Module (src/data/)"),
        ("11", "Communication Module (src/communication/)"),
        ("12", "Main Application (main.py)"),
        ("13", "Calibration System"),
        ("14", "EEG Trigger System"),
        ("15", "Data Output Formats"),
        ("16", "Keyboard Controls & Interaction"),
        ("17", "Command-Line Interface"),
        ("18", "Testing"),
        ("19", "Key Design Decisions"),
        ("20", "Class & Function Reference Summary"),
    ]
    for num, title in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(10, 7, num + ".")
        pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")

    # =========================================================================
    # 1. PROJECT OVERVIEW
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("1", "Project Overview")

    pdf.body_text(
        "The P300 BCI Game is a Brain-Computer Interface (BCI) application developed as an undergraduate "
        "capstone project at Salem State University. It allows a user to control a maze character using only "
        "brain signals - specifically, the P300 event-related potential (ERP) elicited by visual stimuli."
    )

    pdf.section_title("What is the P300 ERP?")
    pdf.body_text(
        "The P300 is a positive voltage deflection in EEG signals that occurs approximately 300 milliseconds "
        "after the presentation of a rare or attended stimulus. In this game, four directional arrows (Up, Down, "
        "Left, Right) flash one at a time. When the user mentally focuses on one specific arrow, their brain "
        "produces a stronger P300 response each time that arrow flashes compared to the non-attended arrows. A "
        "classifier (implemented externally in MATLAB) analyzes these EEG responses to determine which direction "
        "the user intends to move."
    )

    pdf.section_title("Purpose")
    pdf.bullet_point("Provide a gamified interface for P300 BCI research and data collection")
    pdf.bullet_point("Collect precisely-timed EEG trigger data for MATLAB-based P300 classification")
    pdf.bullet_point("Run calibration sessions where users attend to each arrow direction sequentially")
    pdf.bullet_point("Allow gameplay through BCI selection, simulated selection, or manual keyboard input")
    pdf.bullet_point("Support both single-monitor development and dual-monitor lab setups (EEG + game display)")

    pdf.section_title("Key Research References")
    pdf.bullet_point("Farwell & Donchin (1988): Original P300 speller paradigm")
    pdf.bullet_point("Ron-Angevin et al. (2019): Speller size optimization")
    pdf.bullet_point("Takano et al. (2011): Green/Blue chromatic scheme for BCI comfort")

    # =========================================================================
    # 2. VISUAL DESCRIPTION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("2", "How the Game Looks (Visual Description)")

    pdf.section_title("Overall Appearance")
    pdf.body_text(
        "The game renders at an internal design resolution of 3072x1920 pixels and then scales to fit "
        "the actual window size (maintaining aspect ratio with black letterboxing). The overall visual "
        "style is intentionally dark and muted - nearly everything is rendered in grayscale. This design "
        "choice is critical: the game elements must not visually compete with the flashing arrow stimuli, "
        "because the arrows need to produce the strongest possible P300 response in the user's EEG signal."
    )

    pdf.section_title("Screen Layout")
    pdf.body_text(
        "The screen is divided into two overlapping layers:"
    )
    pdf.bullet_point(
        "Background Layer (Maze): A 10x6 cell grid fills the entire screen. Each cell is approximately "
        "307x320 pixels at design resolution. The maze uses pixel art textures - walls are rendered with "
        "a brick pattern and floor tiles have an alternating checkerboard pattern. All maze colors are very "
        "dark grayscale (walls ~RGB(30,30,30), paths ~RGB(18,18,18))."
    )
    pdf.bullet_point(
        "Foreground Layer (Arrow Panel): In the center of the screen is a solid black rectangular panel "
        "(approximately 1190x1190 pixels) that sits on top of the maze. Four arrow triangles are placed in "
        "a cross pattern within this panel - Up arrow at top-center, Down arrow at bottom-center, Left arrow "
        "at center-left, and Right arrow at center-right. Each arrow is offset 475 pixels from the screen "
        "center. The arrows are gray triangles (RGB 128,128,128) when idle and flash to bright white "
        "(RGB 255,255,255) during stimulus presentation."
    )

    pdf.section_title("The Maze")
    pdf.body_text(
        "The maze is generated as an open corridor (no internal walls) where the player can move freely "
        "to any cell that is not occupied by the arrow panel. The maze uses pixel art textures:"
    )
    pdf.bullet_point("Brick wall texture: Small (~4px) pixel blocks with mortar lines, highlight/shadow edges")
    pdf.bullet_point("Floor tile texture: 3x3 alternating light/dark tiles per cell with pixel art shading")
    pdf.bullet_point("The maze has a 'hole' in the center where the arrow panel sits - those cells are simply not rendered")

    pdf.section_title("The Player Character")
    pdf.body_text(
        "The player is represented by a Viking character sprite loaded from 'assets/viking_transparent.png'. "
        "The sprite fills an entire maze cell and is rendered in muted grayscale (darkened via a dullness "
        "factor). The Viking sprite flips horizontally when moving left. Movement between cells is smoothly "
        "animated with quadratic ease-out interpolation over 200ms."
    )

    pdf.section_title("Collectible Items")
    pdf.body_text(
        "The game spawns 5 donut collectibles per level, loaded from 'assets/donut.png'. Donuts are rendered "
        "at 67.5% of cell size, centered within their cell, and apply the same grayscale dullness factor as "
        "other game elements. Each donut is worth 50 points. The player collects them by moving onto the same cell."
    )

    pdf.section_title("Scoreboard")
    pdf.body_text(
        "A pixel art scoreboard is displayed in the top-right corner of the maze, spanning 2 cells wide by "
        "1 cell tall. It uses a pre-made image ('assets/scoreboard.png') as the background with chalk-style "
        "pixel font text overlaid showing SCORE, ITEMS (collected/total), and LVL (level number). The pixel "
        "font is a custom 5x7 pixel character set rendered programmatically."
    )

    pdf.section_title("Arrow Stimuli (The BCI Interface)")
    pdf.body_text(
        "The four arrows are isosceles triangles (150x150px bounding box, with 150px length and 150px base). "
        "Each arrow has a 200x200px panel behind it. The entire arrow cluster sits within a centered black "
        "rectangle. Arrows have pixel art styling with highlight and shadow edges for a 3D effect."
    )
    pdf.body_text(
        "Three color schemes are available (all grayscale-based):"
    )
    pdf.bullet_point("GRAY_WHITE (default): Gray idle (128,128,128) -> White flash (255,255,255)")
    pdf.bullet_point("GREEN_BLUE: Dark gray idle (80,80,80) -> White flash (255,255,255)")
    pdf.bullet_point("INVERTED: Light gray idle (200,200,200) -> White flash (255,255,255)")

    pdf.section_title("Dullness System")
    pdf.body_text(
        "A 'dullness' setting (1-5) controls how dim the game elements appear. At dullness 5 (default), "
        "game elements are at their base brightness. At dullness 1, all game colors are multiplied by 0.2 "
        "(20%), making the maze nearly invisible. This ensures the arrows always have maximum visual contrast. "
        "Dullness is adjustable via the settings panel."
    )

    pdf.section_title("Settings Panel")
    pdf.body_text(
        "Pressing 'S' opens a pixel art styled settings overlay panel with sliders, dropdowns, and buttons. "
        "It allows runtime adjustment of: Flash Duration (50-200ms), ISI (50-200ms), Number of Sequences "
        "(1-30), Color Scheme selection, and Dullness level. The panel uses a retro pixel art style with "
        "inset rectangles and custom-rendered controls."
    )

    pdf.section_title("Calibration Visuals")
    pdf.body_text(
        "During calibration, the system cycles through 4 phases (one for each arrow direction). In the "
        "INSTRUCTION stage, the arrow panel area shows text like 'ATTEND UP' centered on screen. In the "
        "FLASHING stage, arrows flash one at a time. In the BREAK stage, the panel area is cleared to the "
        "background color."
    )

    pdf.section_title("Debug Overlay")
    pdf.body_text(
        "When enabled (press 'D'), a semi-transparent debug panel appears in the top-right corner showing: "
        "FPS, current state, progress percentage, flash timing parameters, window/scale info, calibration "
        "phase info, and last selection result."
    )

    # =========================================================================
    # 3. TECHNOLOGY STACK
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("3", "Technology Stack & Dependencies")

    pdf.section_title("Programming Language")
    pdf.body_text(
        "The entire application is written in Python 3.13 (developed on Python 3.13.7). The codebase uses "
        "modern Python features including dataclasses, type hints, enums, f-strings, and pathlib."
    )

    pdf.section_title("Core Dependencies")
    widths = [40, 25, 125]
    pdf.table_row(["Library", "Version", "Purpose"], widths, bold=True, fill=True)
    pdf.table_row(["pygame", ">=2.5.0", "Game engine: window management, rendering, event handling, display"], widths)
    pdf.table_row(["numpy", ">=1.24.0", "Numerical operations (available but lightly used currently)"], widths)
    pdf.table_row(["scipy", ">=1.10.0", "MATLAB communication support (optional)"], widths)
    pdf.ln(3)

    pdf.section_title("Optional Dependencies")
    widths = [40, 25, 125]
    pdf.table_row(["Library", "Version", "Purpose"], widths, bold=True, fill=True)
    pdf.table_row(["pylsl", ">=1.16.0", "Lab Streaming Layer for real-time EEG trigger streaming"], widths)
    pdf.ln(3)

    pdf.section_title("Development Dependencies")
    widths = [40, 25, 125]
    pdf.table_row(["Library", "Version", "Purpose"], widths, bold=True, fill=True)
    pdf.table_row(["pytest", ">=7.0.0", "Testing framework"], widths)
    pdf.table_row(["pytest-cov", ">=4.0.0", "Code coverage for tests"], widths)
    pdf.table_row(["black", ">=23.0.0", "Code formatter"], widths)
    pdf.table_row(["flake8", ">=6.0.0", "Linter"], widths)
    pdf.ln(3)

    pdf.section_title("Why Pygame?")
    pdf.body_text(
        "Pygame was chosen over PsychoPy (which is more common in neuroscience research) for several reasons: "
        "(1) simpler learning curve for the capstone team, (2) adequate timing precision when combined with "
        "time.perf_counter() busy-wait loops (sub-millisecond accuracy), (3) full control over rendering "
        "pipeline, (4) easier game mechanics implementation. The tradeoff is that Pygame does not have "
        "built-in support for precise display synchronization (no photodiode triggers, no frame-locked timing). "
        "The application compensates by using high-resolution timers and pre-scheduled event queues."
    )

    pdf.section_title("External Processing Pipeline")
    pdf.body_text(
        "The Python game itself does NOT classify EEG signals. The data flow is:"
    )
    pdf.bullet_point("EEG recording hardware: g.tec amplifier records brain signals")
    pdf.bullet_point("Python game writes trigger timestamps to text files (synchronized with flashes)")
    pdf.bullet_point("MATLAB reads trigger files and EEG data, runs P300 classifier")
    pdf.bullet_point("Classifier output (selected direction) is fed back into the game")
    pdf.body_text(
        "Currently, the MATLAB bridge is not fully implemented (the communication/ module is a placeholder). "
        "The game supports simulated BCI selection via keyboard (keys 1-4) for testing."
    )

    # =========================================================================
    # 4. PROJECT STRUCTURE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("4", "Project Structure & File Layout")

    pdf.code_block(
        "p300-bci-game/\n"
        "|-- main.py                    # Application entry point\n"
        "|-- config.py                  # Central configuration (dataclasses)\n"
        "|-- requirements.txt           # Python dependencies\n"
        "|-- README.md                  # Brief project description\n"
        "|-- LICENSE                    # Project license\n"
        "|\n"
        "|-- src/                       # Source code package\n"
        "|   |-- __init__.py            # Package init (v0.1.0)\n"
        "|   |-- stimulus/              # Arrow stimulus presentation\n"
        "|   |   |-- __init__.py\n"
        "|   |   |-- arrow_manager.py   # Main stimulus coordinator\n"
        "|   |   |-- arrow_renderer.py  # Arrow drawing & layout\n"
        "|   |   |-- timing_controller.py # Precise flash timing\n"
        "|   |-- game/                  # Game mechanics\n"
        "|   |   |-- __init__.py\n"
        "|   |   |-- game_manager.py    # Central game coordinator\n"
        "|   |   |-- maze.py            # Maze generation & structure\n"
        "|   |   |-- player.py          # Player entity & movement\n"
        "|   |   |-- collectible.py     # Collectible items & scoring\n"
        "|   |   |-- renderer.py        # Pixel art game renderer\n"
        "|   |   |-- auto_play_controller.py  # Automated testing\n"
        "|   |-- ui/                    # User interface\n"
        "|   |   |-- __init__.py\n"
        "|   |   |-- settings_panel.py  # Runtime settings UI\n"
        "|   |   |-- mode_selector.py   # Startup mode selection\n"
        "|   |-- data/                  # Data recording\n"
        "|   |   |-- session_logger.py  # Session log writer\n"
        "|   |-- communication/         # External communication\n"
        "|       |-- __init__.py        # Placeholder (MATLAB bridge)\n"
        "|\n"
        "|-- assets/                    # Image assets\n"
        "|   |-- viking_transparent.png # Player character sprite\n"
        "|   |-- donut.png              # Collectible item sprite\n"
        "|   |-- scoreboard.png         # Scoreboard background\n"
        "|   |-- coffee_cup.png         # Unused alt collectible\n"
        "|   |-- floppy_disk.png        # Unused alt collectible\n"
        "|\n"
        "|-- data/                      # Output data\n"
        "|   |-- sessions/              # Session & trigger logs\n"
        "|   |-- calibration/           # Calibration data (empty)\n"
        "|\n"
        "|-- tests/                     # Test suite\n"
        "|   |-- test_config.py         # Config validation tests\n"
        "|   |-- test_timing.py         # Timer precision tests\n"
        "|\n"
        "|-- docs/                      # Documentation\n"
        "|   |-- ARCHITECTURE.md        # Architecture overview\n"
        "|\n"
        "|-- matlab/                    # MATLAB scripts (empty)\n"
    )

    # =========================================================================
    # 5. ARCHITECTURE & DATA FLOW
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("5", "Architecture & Data Flow")

    pdf.section_title("Module Dependency Graph")
    pdf.code_block(
        "main.py (Application)\n"
        "  |-- config.py (Config)\n"
        "  |-- ArrowManager (src/stimulus/)\n"
        "  |     |-- ArrowRenderer   - Draws arrow triangles\n"
        "  |     |-- TimingController - Schedules flash events\n"
        "  |     |-- TriggerManager  - Writes EEG trigger files\n"
        "  |-- GameManager (src/game/)\n"
        "  |     |-- Maze            - Grid generation\n"
        "  |     |-- Player          - Position & movement\n"
        "  |     |-- CollectibleManager - Item spawning & scoring\n"
        "  |     |-- GameRenderer    - Pixel art rendering\n"
        "  |-- SettingsPanel (src/ui/)\n"
        "  |-- SessionLogger (src/data/)"
    )

    pdf.section_title("Data Flow During Normal Gameplay")
    pdf.body_text(
        "1. The user presses SPACE to start a calibration/selection run.\n"
        "2. ArrowManager activates TimingController, which pre-schedules all flash events.\n"
        "3. Each frame, TimingController checks time.perf_counter() against the schedule.\n"
        "4. When a flash event fires: ArrowRenderer updates the visual, TriggerManager writes a timestamp "
        "to the trigger file, and SessionLogger records the flash for the session log.\n"
        "5. After all sequences complete, the system enters PROCESSING state waiting for classifier input.\n"
        "6. When a direction is selected (via MATLAB classifier or keyboard simulation), the GameManager "
        "moves the Player in that direction on the Maze grid.\n"
        "7. The GameRenderer draws the updated maze, player, and collectibles."
    )

    pdf.section_title("Data Flow During Calibration")
    pdf.body_text(
        "Calibration differs from normal selection in that the user is instructed which arrow to attend:\n"
        "1. A random order of 4 directions is generated (e.g., LEFT, UP, RIGHT, DOWN).\n"
        "2. For each direction: INSTRUCTION phase (2s) shows 'ATTEND [direction]', FLASHING phase runs "
        "all sequences with randomized flash order, BREAK phase (2s) shows neutral screen.\n"
        "3. TriggerManager logs each flash with the current attended target for classifier training.\n"
        "4. SessionLogger writes all flash events with timestamps for offline analysis."
    )

    pdf.section_title("Rendering Pipeline")
    pdf.body_text(
        "All rendering goes through a resolution-independent pipeline:\n"
        "1. All game components draw to a render_surface at design resolution (3072x1920).\n"
        "2. Rendering order: Game (maze, collectibles, player) -> Arrows (or calibration) -> Debug overlay -> Settings panel.\n"
        "3. The render_surface is scaled via pygame.transform.smoothscale() to fit the window.\n"
        "4. The scaled surface is blitted centered on the screen with black letterbox bars.\n"
        "5. Mouse coordinates are reverse-transformed from window space to design space for UI interaction."
    )

    # =========================================================================
    # 6. CONFIGURATION SYSTEM
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("6", "Configuration System")

    pdf.body_text(
        "All configuration is centralized in config.py using Python dataclasses. The Config class aggregates "
        "six sub-configurations:"
    )

    pdf.section_title("DisplayConfig")
    pdf.key_value("width", "1920 (default window width)")
    pdf.key_value("height", "1080 (default window height)")
    pdf.key_value("fullscreen", "False")
    pdf.key_value("fps", "60 (target frame rate)")
    pdf.key_value("vsync", "True")
    pdf.key_value("background_color", "(10, 10, 10) - nearly black")

    pdf.section_title("TimingConfig")
    pdf.key_value("flash_duration_ms", "100 (how long arrow stays highlighted)")
    pdf.key_value("isi_ms", "125 (inter-stimulus interval between flashes)")
    pdf.key_value("soa_ms", "225 (computed: flash_duration + isi)")
    pdf.key_value("flash_rate_hz", "4.44 Hz (computed: 1000/soa)")
    pdf.key_value("num_sequences", "10 (repetitions per selection)")
    pdf.key_value("inter_sequence_pause_ms", "200")
    pdf.key_value("flash_pattern", "RANDOM (randomized order each sequence)")
    pdf.key_value("feedback_duration_ms", "500")

    pdf.section_title("ArrowConfig")
    pdf.key_value("size", "150 px (arrow bounding box)")
    pdf.key_value("triangle_length", "150 px")
    pdf.key_value("triangle_base", "150 px")
    pdf.key_value("color_scheme", "GRAY_WHITE")
    pdf.key_value("idle_color", "(128,128,128) - medium gray")
    pdf.key_value("flash_color", "(255,255,255) - bright white")
    pdf.key_value("panel_size", "200 px (per arrow panel)")
    pdf.key_value("panel_alpha", "153 (~60% opacity)")
    pdf.key_value("glow_thickness", "20 px (border when flashing)")

    pdf.section_title("LayoutConfig")
    pdf.key_value("horizontal_offset", "475 px from center (Left/Right arrows)")
    pdf.key_value("vertical_offset", "475 px from center (Up/Down arrows)")
    pdf.key_value("keepout_margin", "50 px (spacing from panels to game)")

    pdf.section_title("GameConfig")
    pdf.key_value("wall_color", "(40,40,40) - dark gray")
    pdf.key_value("path_color", "(25,25,25) - very dark gray")
    pdf.key_value("player_color", "(70,70,70)")
    pdf.key_value("cell_size", "40 px (base)")
    pdf.key_value("player_size", "30 px")
    pdf.key_value("move_duration_ms", "300")
    pdf.key_value("dullness", "5 (1=very dim, 5=normal)")

    pdf.section_title("TriggerConfig")
    pdf.key_value("enabled", "True")
    pdf.key_value("method", '"file" (alternatives: "lsl", "serial")')
    pdf.key_value("trigger_file", "data/sessions/triggers.txt")
    pdf.key_value("Trigger codes", "1=trial_start, 2=trial_end, 10-13=flash directions, 20=selection")

    # =========================================================================
    # 7. STIMULUS MODULE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("7", "Stimulus Module (src/stimulus/)")

    pdf.body_text(
        "The stimulus module handles the P300 arrow presentation system. This is the most scientifically "
        "critical part of the application - precise timing and consistent visuals are essential for reliable "
        "P300 evoked potentials."
    )

    pdf.section_title("ArrowRenderer (arrow_renderer.py)")
    pdf.body_text(
        "Handles all visual aspects of the arrow stimuli. Pre-renders arrow surfaces for performance "
        "(idle and flash variants for each direction). Creates a solid black background panel. "
        "Supports runtime color scheme switching."
    )
    pdf.bullet_point("Creates isosceles triangle polygons for each direction (Up/Down/Left/Right)")
    pdf.bullet_point("Adds pixel art styling: highlight edges on top-left, shadow edges on bottom-right")
    pdf.bullet_point("The panel is a full opaque black rectangle (expanded 20px beyond arrow bounds)")
    pdf.bullet_point("draw(screen, flash_states) is called every frame with a dict of {Direction: bool}")

    pdf.section_title("TimingController (timing_controller.py)")
    pdf.body_text(
        "Provides sub-millisecond timing precision for stimulus presentation. This is critical because "
        "P300 BCI requires timing jitter < 10ms for reliable ERP averaging."
    )
    pdf.bullet_point("Uses time.perf_counter() for high-resolution timing")
    pdf.bullet_point("Pre-schedules ALL flash events before starting (minimizes runtime overhead)")
    pdf.bullet_point("Each event is a FlashEvent dataclass: time_ms, direction, is_on, sequence, index")
    pdf.bullet_point("update() processes all due events in a single frame (handles missed frames)")
    pdf.bullet_point("Tracks timing statistics: mean error, max error, acceptability check (<10ms)")
    pdf.bullet_point("Supports RANDOM and SEQUENTIAL flash patterns")

    pdf.section_title("ArrowManager (arrow_manager.py)")
    pdf.body_text(
        "Coordinates the renderer, timing, and trigger systems into a unified interface. Manages "
        "the BCI selection state machine."
    )
    pdf.bullet_point("SelectionState: IDLE -> FLASHING -> PROCESSING -> FEEDBACK -> COMPLETE -> IDLE")
    pdf.bullet_point("start_selection(): Resets state, starts trigger session, starts timing")
    pdf.bullet_point("update(): Drives TimingController, manages state transitions")
    pdf.bullet_point("simulate_selection(direction): Bypass for testing without EEG")
    pdf.bullet_point("Contains TriggerManager class (EEG synchronization)")

    # =========================================================================
    # 8. GAME MODULE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("8", "Game Module (src/game/)")

    pdf.section_title("Maze (maze.py)")
    pdf.body_text(
        "Handles maze generation and grid storage. Supports two generation modes:"
    )
    pdf.bullet_point(
        "Recursive backtracking (use_corridors=False): Creates perfect mazes with exactly one path "
        "between any two points. Uses iterative stack to avoid recursion limits. Requires odd grid dimensions."
    )
    pdf.bullet_point(
        "Corridor mode (use_corridors=True): Creates a fully open playing field with no internal walls. "
        "Used when cell_size >= 128px. The player can move freely to any non-forbidden cell."
    )
    pdf.body_text(
        "The maze supports a 'forbidden zone' in the center where the arrow panel sits. This zone can be "
        "either rectangular or plus-shaped. Cells in the forbidden zone are treated as walls and never "
        "rendered. Grid cells are of type CellType: WALL(0), PATH(1), START(2), or GOAL(3)."
    )
    pdf.body_text(
        "Current configuration: 10 columns x 6 rows at ~307px cell size, corridor mode, producing a "
        "fully open field with a central hole for the arrow panel."
    )

    pdf.section_title("Player (player.py)")
    pdf.body_text(
        "Manages player state and grid-based movement with smooth animation."
    )
    pdf.bullet_point("Grid position tracked as integer (x, y) coordinates")
    pdf.bullet_point("Movement validated via is_walkable callback before execution")
    pdf.bullet_point("Smooth interpolation between cells using quadratic ease-out: t' = 1 - (1-t)^2")
    pdf.bullet_point("render_position property returns fractional coordinates for smooth visual movement")
    pdf.bullet_point("Default move duration: 200ms per cell")
    pdf.bullet_point("State machine: IDLE <-> MOVING (cannot accept new input while moving)")

    pdf.section_title("CollectibleManager (collectible.py)")
    pdf.body_text(
        "Manages spawning, collection, and scoring of items."
    )
    pdf.bullet_point("Currently uses only STAR type (donut sprite), worth 50 points each")
    pdf.bullet_point("5 donuts spawned per level at random path cells (excluding used positions)")
    pdf.bullet_point("Collection detected when player moves onto same grid cell")
    pdf.bullet_point("all_collected() returns True when all items are picked up -> triggers level complete")

    pdf.section_title("GameRenderer (renderer.py)")
    pdf.body_text(
        "The most visually complex module. Renders all game elements in a retro pixel art style."
    )
    pdf.bullet_point("Pixel size: max(2, cell_size // 40) - approximately 4px pixels for 160px cells")
    pdf.bullet_point("Brick wall texture: Row-offset bricks with mortar, highlight, shadow, and noise")
    pdf.bullet_point("Floor tile texture: 3x3 alternating checkerboard per cell with edge shading")
    pdf.bullet_point("Viking sprite: Loaded from PNG, scaled to cell size, with dullness applied")
    pdf.bullet_point("Donut sprite: Loaded from PNG, cropped, scaled to 67.5% of cell, dullness applied")
    pdf.bullet_point("Scoreboard: PNG background with programmatic 5x7 pixel font overlay (chalk style)")
    pdf.bullet_point("Maze caching: Full maze rendered to a surface once, cached until maze changes")
    pdf.bullet_point("Dullness: All colors multiplied by (dullness/5.0) factor, applied to both colors and sprites")

    pdf.section_title("GameManager (game_manager.py)")
    pdf.body_text(
        "Central game coordinator. Manages game state, level generation, and integrates all game subsystems."
    )
    pdf.bullet_point("GameState: READY -> PLAYING -> LEVEL_COMPLETE or GAME_OVER")
    pdf.bullet_point("Level generation: Creates Maze, places Player at start, spawns CollectibleManager items")
    pdf.bullet_point("move_player(direction): Validates and executes player movement, checks collection")
    pdf.bullet_point("Level progression: When all items collected, advances to next level with new maze")
    pdf.bullet_point("Configurable via GameManagerConfig: maze dimensions, collectible counts, cell size")

    pdf.section_title("AutoPlayController (auto_play_controller.py)")
    pdf.body_text(
        "Automated testing utility that generates random movement directions. Not currently wired into "
        "main.py but available for automated testing. Supports configurable delay, valid-move preference, "
        "backtracking avoidance, and optional file logging."
    )

    # =========================================================================
    # 9. UI MODULE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("9", "UI Module (src/ui/)")

    pdf.section_title("SettingsPanel (settings_panel.py)")
    pdf.body_text(
        "Runtime settings overlay with pixel art styled controls. Allows adjusting BCI parameters "
        "without restarting the application."
    )
    pdf.bullet_point("PixelSlider: Horizontal slider for numeric values (flash duration, ISI, sequences, dullness)")
    pdf.bullet_point("PixelDropdown: Selection control for color scheme and flash pattern")
    pdf.bullet_point("PixelButton: Apply/Cancel buttons")
    pdf.bullet_point("SettingsValues dataclass: flash_duration_ms, isi_ms, num_sequences, color_scheme, dullness")
    pdf.bullet_point("from_config() and apply_to_config() methods for syncing with Config object")
    pdf.bullet_point("Callbacks: on_apply triggers ArrowManager reinitialization, on_cancel restores previous values")

    pdf.section_title("ModeSelector (mode_selector.py)")
    pdf.body_text(
        "Startup mode selection UI (not currently wired into main.py). Defines three application modes:"
    )
    pdf.bullet_point("DATA_COLLECTION: Calibration/data recording mode")
    pdf.bullet_point("GAME: Full game mode with BCI control")
    pdf.bullet_point("GAME_TESTING: Game mode with keyboard/simulated BCI")

    # =========================================================================
    # 10. DATA MODULE
    # =========================================================================
    pdf.chapter_title("10", "Data Module (src/data/)")

    pdf.section_title("SessionLogger (session_logger.py)")
    pdf.body_text(
        "Records complete BCI session data for offline analysis. Each session produces a timestamped "
        "text file in data/sessions/."
    )
    pdf.bullet_point("start_session(): Creates new session with timing parameters")
    pdf.bullet_point("log_flash_start(direction, sequence, timestamp_ms): Records flash onset")
    pdf.bullet_point("log_flash_end(direction, sequence, timestamp_ms): Records flash offset")
    pdf.bullet_point("end_session(): Calculates statistics and saves to file")
    pdf.bullet_point("cancel_session(): Discards unsaved data")
    pdf.bullet_point("Output filename format: session_YYYYMMDD_HHMMSS_fff.txt")

    # =========================================================================
    # 11. COMMUNICATION MODULE
    # =========================================================================
    pdf.chapter_title("11", "Communication Module (src/communication/)")

    pdf.body_text(
        "Currently a placeholder module. The __init__.py has commented-out exports for TriggerManager "
        "and MatlabBridge. The TriggerManager implementation actually lives inside arrow_manager.py. "
        "The matlab/ directory at the project root is empty. Future work would implement a MATLAB bridge "
        "for real-time classification result exchange, potentially via file polling, LSL, or socket communication."
    )

    # =========================================================================
    # 12. MAIN APPLICATION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("12", "Main Application (main.py)")

    pdf.body_text(
        "The Application class in main.py is the central orchestrator. It initializes Pygame, creates all "
        "subsystems, and runs the main game loop."
    )

    pdf.section_title("Initialization Sequence")
    pdf.body_text(
        "1. Parse command-line arguments (fullscreen, debug, width, height, sequences, display)\n"
        "2. Create Config object and apply CLI overrides\n"
        "3. Initialize Pygame with DOUBLEBUF flag (+ FULLSCREEN if requested)\n"
        "4. Auto-detect displays (uses second monitor if available for lab setups)\n"
        "5. Create render surface at design resolution (3072x1920)\n"
        "6. Calculate scaling factor and letterbox offsets\n"
        "7. Initialize ArrowManager -> SettingsPanel -> GameManager -> SessionLogger\n"
        "8. Print startup info to console"
    )

    pdf.section_title("Main Loop")
    pdf.body_text(
        "The main loop runs at 60 FPS and follows a standard game loop pattern:\n"
        "1. _handle_events(): Process Pygame events, transform mouse coordinates, delegate to subsystems\n"
        "2. _update(): Update calibration state machine OR arrow manager, update game manager\n"
        "3. _draw(): Clear surface, draw game, draw arrows, draw debug, draw settings, scale to window\n"
        "4. clock.tick(fps): Frame rate limiting"
    )

    pdf.section_title("Resolution Independence")
    pdf.body_text(
        "The game always renders internally at 3072x1920 (design resolution) regardless of window size. "
        "The actual window can be any size - the game scales proportionally with letterboxing. Mouse "
        "coordinates are reverse-transformed from window space to design space for accurate UI interaction. "
        "Default window size is 3072x1920 but can be overridden via --width and --height flags."
    )

    # =========================================================================
    # 13. CALIBRATION SYSTEM
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("13", "Calibration System")

    pdf.body_text(
        "The calibration system is the primary data collection mechanism. It cycles through all four "
        "arrow directions, instructing the user to attend to each one while flashing all arrows. This "
        "generates labeled EEG data for training the P300 classifier."
    )

    pdf.section_title("Calibration Run Structure")
    pdf.body_text(
        "One calibration run consists of 4 phases (one per direction) in random order:"
    )
    pdf.code_block(
        "For each of 4 directions (randomized order):\n"
        "  1. INSTRUCTION stage (2000ms):\n"
        "     - Display 'ATTEND [DIRECTION]' text\n"
        "     - User fixates on the indicated arrow\n"
        "  2. FLASHING stage:\n"
        "     - Flash all 4 arrows in random order\n"
        "     - Repeat for num_sequences (default: 10)\n"
        "     - Each flash: 100ms ON, 125ms ISI\n"
        "     - Total per direction: 10 seq x 4 arrows x 225ms = 9000ms\n"
        "  3. BREAK stage (2000ms):\n"
        "     - Neutral screen (no arrows visible)\n"
        "     - User can relax before next direction"
    )

    pdf.section_title("Timing Details")
    pdf.body_text(
        "The calibration manages its own flash timing (independent of TimingController) using "
        "time.perf_counter() for precision. The calibration_flash_plan is a flat list of Direction "
        "values representing every flash in order. Flash timing uses busy-wait comparison against "
        "perf_counter timestamps."
    )

    pdf.section_title("Data Generated")
    pdf.body_text(
        "Each calibration run generates two synchronized output files:\n"
        "1. Session log (via SessionLogger): session_YYYYMMDD_HHMMSS_fff.txt with all flash events\n"
        "2. Trigger log (via TriggerManager): triggers_YYYYMMDD_HHMMSS.txt with trigger codes and "
        "the currently attended target direction for each event"
    )

    # =========================================================================
    # 14. EEG TRIGGER SYSTEM
    # =========================================================================
    pdf.chapter_title("14", "EEG Trigger System")

    pdf.body_text(
        "The TriggerManager (defined in arrow_manager.py) handles synchronization between the game's "
        "visual stimuli and external EEG recording."
    )

    pdf.section_title("Trigger Codes")
    widths = [25, 40, 125]
    pdf.table_row(["Code", "Label", "Description"], widths, bold=True, fill=True)
    pdf.table_row(["1", "trial_start", "Marks beginning of a selection/calibration trial"], widths)
    pdf.table_row(["2", "trial_end", "Marks end of a trial"], widths)
    pdf.table_row(["10", "flash_up", "Up arrow flash onset"], widths)
    pdf.table_row(["11", "flash_down", "Down arrow flash onset"], widths)
    pdf.table_row(["12", "flash_left", "Left arrow flash onset"], widths)
    pdf.table_row(["13", "flash_right", "Right arrow flash onset"], widths)
    pdf.table_row(["20", "selection_*", "Final direction selection made"], widths)
    pdf.ln(3)

    pdf.section_title("Output Methods")
    pdf.bullet_point("file (current): Writes to timestamped text files in data/sessions/")
    pdf.bullet_point("lsl (planned): Lab Streaming Layer for real-time streaming")
    pdf.bullet_point("serial (planned): Serial port for hardware trigger boxes")

    # =========================================================================
    # 15. DATA OUTPUT FORMATS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("15", "Data Output Formats")

    pdf.section_title("Session Log Format")
    pdf.body_text("Files: data/sessions/session_YYYYMMDD_HHMMSS_fff.txt")
    pdf.code_block(
        "======================================================================\n"
        "P300 BCI SESSION LOG\n"
        "======================================================================\n"
        "SESSION INFORMATION:\n"
        "  Session ID, Start Time, End Time, Duration\n"
        "\n"
        "SESSION PARAMETERS:\n"
        "  Flash Duration, ISI, SOA, Num Sequences, Flash Pattern, Color Scheme\n"
        "\n"
        "FLASH EVENTS:\n"
        "  Timestamp_ms | Absolute_Time | Direction | Sequence | Event_Type\n"
        "  0.000        | HH:MM:SS.fff  | up        | 0        | flash_start\n"
        "  100.150      | HH:MM:SS.fff  | up        | 0        | flash_end\n"
        "  ..."
    )

    pdf.section_title("Trigger Log Format")
    pdf.body_text("Files: data/sessions/triggers_YYYYMMDD_HHMMSS.txt")
    pdf.code_block(
        "======================================================================\n"
        "P300 BCI TRIGGER LOG\n"
        "======================================================================\n"
        "\n"
        "Session started: YYYY-MM-DD HH:MM:SS.fff\n"
        "\n"
        "SESSION PARAMETERS:\n"
        "  Flash Duration, ISI, Num Sequences\n"
        "\n"
        "TRIGGER EVENTS:\n"
        "  Format: timestamp_ms, label, current_target\n"
        "\n"
        "  0.000, trial_start, UP\n"
        "  5.123, flash_left, UP\n"
        "  230.456, flash_down, UP\n"
        "  ...\n"
        "  9050.789, trial_end, UP\n"
        "\n"
        "======================================================================\n"
        "Session ended: YYYY-MM-DD HH:MM:SS.fff"
    )

    # =========================================================================
    # 16. KEYBOARD CONTROLS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("16", "Keyboard Controls & Interaction")

    widths = [35, 155]
    pdf.table_row(["Key", "Action"], widths, bold=True, fill=True)
    pdf.table_row(["SPACE", "Start one full calibration run (4 directions x N sequences)"], widths)
    pdf.table_row(["S", "Toggle settings panel (cannot open during active calibration)"], widths)
    pdf.table_row(["D", "Toggle debug overlay (FPS, state, timing info)"], widths)
    pdf.table_row(["R", "Restart current level (regenerate maze and collectibles)"], widths)
    pdf.table_row(["N", "Skip to next level (for testing)"], widths)
    pdf.table_row(["1", "Simulate BCI selection: UP direction"], widths)
    pdf.table_row(["2", "Simulate BCI selection: DOWN direction"], widths)
    pdf.table_row(["3", "Simulate BCI selection: LEFT direction"], widths)
    pdf.table_row(["4", "Simulate BCI selection: RIGHT direction"], widths)
    pdf.table_row(["Arrow Up", "Manual movement: move player up (testing)"], widths)
    pdf.table_row(["Arrow Down", "Manual movement: move player down (testing)"], widths)
    pdf.table_row(["Arrow Left", "Manual movement: move player left (testing)"], widths)
    pdf.table_row(["Arrow Right", "Manual movement: move player right (testing)"], widths)
    pdf.table_row(["ESC", "Quit application (saves/cancels active sessions)"], widths)
    pdf.ln(3)

    # =========================================================================
    # 17. CLI
    # =========================================================================
    pdf.chapter_title("17", "Command-Line Interface")

    pdf.body_text("The application accepts the following command-line arguments:")
    pdf.code_block(
        "python main.py [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --fullscreen, -f       Run in fullscreen mode\n"
        "  --debug, -d            Enable debug overlay at startup\n"
        "  --width WIDTH          Window width (default: 3072)\n"
        "  --height HEIGHT        Window height (default: 1920)\n"
        "  --sequences N          Number of sequences per selection\n"
        "  --display INDEX        Display index (0=primary, 1=secondary)\n"
        "                         Auto-selects secondary display if available"
    )

    pdf.body_text(
        "Examples:\n"
        "  python main.py                           # Default 3072x1920 windowed\n"
        "  python main.py --fullscreen              # Fullscreen on auto-detected display\n"
        "  python main.py --width 1920 --height 1080 # Custom window size\n"
        "  python main.py --display 0 -f            # Fullscreen on primary monitor\n"
        "  python main.py --sequences 5 --debug     # 5 sequences with debug overlay"
    )

    # =========================================================================
    # 18. TESTING
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("18", "Testing")

    pdf.body_text(
        "The project includes a test suite in tests/ using pytest."
    )

    pdf.section_title("test_config.py")
    pdf.bullet_point("Validates default configuration values")
    pdf.bullet_point("Tests SOA computation (flash_duration + isi)")
    pdf.bullet_point("Tests Direction enum contains all 4 directions")

    pdf.section_title("test_timing.py")
    pdf.bullet_point("Tests time.perf_counter() resolution (should be sub-millisecond)")
    pdf.bullet_point("Tests busy-wait timing accuracy over short intervals")

    pdf.body_text(
        "Run tests with: pytest tests/ -v\n"
        "Run with coverage: pytest tests/ --cov=src --cov-report=html"
    )

    # =========================================================================
    # 19. KEY DESIGN DECISIONS
    # =========================================================================
    pdf.chapter_title("19", "Key Design Decisions")

    pdf.section_title("1. Pygame over PsychoPy")
    pdf.body_text(
        "PsychoPy is the standard for psychophysics research but has a steep learning curve. Pygame "
        "was chosen for simplicity and adequate timing (verified via test_timing.py). Tradeoff: no "
        "built-in vsync/photodiode support."
    )

    pdf.section_title("2. File-Based Triggers")
    pdf.body_text(
        "Trigger events are written to text files that MATLAB reads. This is the simplest integration "
        "method and avoids complex inter-process communication. Can be upgraded to Lab Streaming Layer "
        "(LSL) for real-time applications."
    )

    pdf.section_title("3. Busy-Wait Timing")
    pdf.body_text(
        "The TimingController uses time.perf_counter() (sub-microsecond resolution) instead of "
        "pygame.time.Clock or sleep(). This achieves sub-millisecond timing precision at the cost of "
        "higher CPU usage. Pre-scheduled events minimize runtime overhead."
    )

    pdf.section_title("4. Grayscale Game Elements")
    pdf.body_text(
        "All game visuals (maze, player, collectibles, UI) are rendered in dark grayscale. This is a "
        "deliberate BCI design choice: the arrow flashes must produce the strongest possible visual "
        "contrast to evoke robust P300 potentials. Colored game elements would compete with stimuli."
    )

    pdf.section_title("5. Resolution-Independent Design")
    pdf.body_text(
        "The game renders at a fixed 3072x1920 design resolution and scales to any window size. This "
        "ensures consistent visual angle of stimuli regardless of display. The design resolution was "
        "chosen to match high-DPI displays common in modern setups."
    )

    pdf.section_title("6. Corridor Mode for Large Cells")
    pdf.body_text(
        "When cells are >= 128px, the maze switches from complex maze generation to open corridor mode. "
        "With only 10x6 cells, a traditional maze would be trivially simple. The open field allows "
        "free movement while keeping the arrow panel hole as the navigation challenge."
    )

    # =========================================================================
    # 20. CLASS & FUNCTION REFERENCE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("20", "Class & Function Reference Summary")

    pdf.section_title("Enums")
    widths = [55, 135]
    pdf.table_row(["Enum", "Values"], widths, bold=True, fill=True)
    pdf.table_row(["Direction", "UP, DOWN, LEFT, RIGHT"], widths)
    pdf.table_row(["ColorScheme", "GRAY_WHITE, GREEN_BLUE, INVERTED"], widths)
    pdf.table_row(["FlashPattern", "RANDOM, SEQUENTIAL"], widths)
    pdf.table_row(["CellType", "WALL(0), PATH(1), START(2), GOAL(3)"], widths)
    pdf.table_row(["PlayerState", "IDLE, MOVING"], widths)
    pdf.table_row(["CollectibleType", "GEM, STAR, POWERUP"], widths)
    pdf.table_row(["SelectionState", "IDLE, FLASHING, PROCESSING, FEEDBACK, COMPLETE"], widths)
    pdf.table_row(["TimerState", "IDLE, RUNNING, PAUSED, COMPLETE"], widths)
    pdf.table_row(["CalibrationStage", "IDLE, INSTRUCTION, FLASHING, BREAK"], widths)
    pdf.table_row(["GameState", "READY, PLAYING, PAUSED, LEVEL_COMPLETE, GAME_OVER"], widths)
    pdf.table_row(["AppMode", "DATA_COLLECTION, GAME, GAME_TESTING"], widths)
    pdf.ln(3)

    pdf.section_title("Key Classes")
    widths = [55, 135]
    pdf.table_row(["Class", "Module & Purpose"], widths, bold=True, fill=True)
    pdf.table_row(["Application", "main.py - Main app controller & game loop"], widths)
    pdf.table_row(["Config", "config.py - Aggregates all configuration"], widths)
    pdf.table_row(["ArrowManager", "stimulus/arrow_manager.py - Stimulus coordinator"], widths)
    pdf.table_row(["ArrowRenderer", "stimulus/arrow_renderer.py - Arrow drawing"], widths)
    pdf.table_row(["TimingController", "stimulus/timing_controller.py - Flash timing"], widths)
    pdf.table_row(["TriggerManager", "stimulus/arrow_manager.py - EEG triggers"], widths)
    pdf.table_row(["GameManager", "game/game_manager.py - Game coordinator"], widths)
    pdf.table_row(["Maze", "game/maze.py - Grid generation & storage"], widths)
    pdf.table_row(["Player", "game/player.py - Player entity"], widths)
    pdf.table_row(["CollectibleManager", "game/collectible.py - Items & scoring"], widths)
    pdf.table_row(["GameRenderer", "game/renderer.py - Pixel art renderer"], widths)
    pdf.table_row(["SettingsPanel", "ui/settings_panel.py - Settings overlay"], widths)
    pdf.table_row(["SessionLogger", "data/session_logger.py - Session recording"], widths)
    pdf.table_row(["AutoPlayController", "game/auto_play_controller.py - Auto testing"], widths)
    pdf.ln(5)

    pdf.section_title("Key Dataclasses")
    widths = [60, 130]
    pdf.table_row(["Dataclass", "Key Fields"], widths, bold=True, fill=True)
    pdf.table_row(["DisplayConfig", "width, height, fullscreen, fps, background_color"], widths)
    pdf.table_row(["TimingConfig", "flash_duration_ms, isi_ms, num_sequences, flash_pattern"], widths)
    pdf.table_row(["ArrowConfig", "size, triangle_length, color_scheme, panel_size"], widths)
    pdf.table_row(["LayoutConfig", "horizontal_offset, vertical_offset, keepout_margin"], widths)
    pdf.table_row(["GameConfig", "wall_color, path_color, cell_size, dullness"], widths)
    pdf.table_row(["TriggerConfig", "enabled, method, trigger_file, FLASH_UP..FLASH_RIGHT"], widths)
    pdf.table_row(["MazeConfig", "width, height, cell_size, seed, use_corridors"], widths)
    pdf.table_row(["PlayerConfig", "move_duration_ms, size_ratio, color"], widths)
    pdf.table_row(["CollectibleConfig", "gem_points, star_points, size_ratio, colors"], widths)
    pdf.table_row(["RenderConfig", "cell_size, dullness, base colors (wall, path, player)"], widths)
    pdf.table_row(["GameManagerConfig", "base_maze_width/height, base_collectibles, cell_size"], widths)
    pdf.table_row(["FlashEvent", "time_ms, direction, is_on, sequence, index"], widths)
    pdf.table_row(["SelectionResult", "direction, confidence, num_sequences, duration_ms"], widths)
    pdf.table_row(["SettingsValues", "flash_duration_ms, isi_ms, num_sequences, color_scheme"], widths)

    # =========================================================================
    # Save
    # =========================================================================
    output_path = "docs/P300_BCI_Game_Documentation.pdf"
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
