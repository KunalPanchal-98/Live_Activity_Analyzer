"""
Modern AI Surveillance Dashboard GUI for Live Activity Analyzer.

Professional desktop surveillance interface built with Tkinter, ttk, and Pillow.
Features:
- Dark surveillance dashboard aesthetic with consistent color palette
- Top navigation & telemetry bar with live FPS and system status
- Multi-page sidebar navigation (Dashboard, Live Monitor, People, Activities,
  Anomalies, Statistics, Events, Camera, Settings, About)
- Real-time metric cards (People, Active Tracks, Objects, FPS, Camera Status,
  Current Activity, Anomalies) with real application values or N/A
- Live video monitor with OpenCV BGR->RGB color correction and aspect-ratio scaling
- Interactive camera state rendering (SEARCHING, CONNECTING, CONNECTED,
  RECONNECTING, DISCONNECTED, NO_CAMERA, ERROR) with fallback controls
- Activity mode selector (Auto Detect, Body Activity, Posture, Movement, Hand Gesture, Face Gesture)
- Real-time event log and tracking tables
- Thread-safe updates from background video processing pipeline
"""

import sys
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from PIL import Image, ImageTk

from app.config_manager import ConfigManager, AppConfig, get_config_manager
from app.camera_manager import CameraState, CameraDeviceInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color Palette Constants (Modern Dark AI Surveillance)
# ---------------------------------------------------------------------------
COLOR_BG_DARK = "#0f1217"        # Base application background
COLOR_SIDEBAR = "#151921"        # Sidebar panel background
COLOR_TOPBAR = "#1a202c"         # Top navigation and telemetry bar
COLOR_PANEL = "#1e2430"          # Card and panel background
COLOR_PANEL_HOVER = "#262d3d"    # Card hover / active row
COLOR_BORDER = "#2d3748"         # Separators and subtle borders
COLOR_ACCENT = "#00b4d8"         # Surveillance cyan / primary accent
COLOR_ACCENT_HOVER = "#0077b6"   # Accent hover / pressed
COLOR_SUCCESS = "#10b981"        # Mint emerald green (online / normal)
COLOR_WARNING = "#f59e0b"        # Amber yellow (warning / alert)
COLOR_DANGER = "#ef4444"         # Crimson red (alarm / error)
COLOR_ORANGE = "#f97316"         # Orange (reconnecting)
COLOR_TEXT_WHITE = "#f8fafc"     # Primary text
COLOR_TEXT_MUTED = "#94a3b8"     # Secondary labels
COLOR_TEXT_DIM = "#64748b"       # Muted placeholders


class MetricCard(ttk.Frame):
    """
    Reusable dashboard metric card displaying:
    - Uppercase category title
    - Prominent bold KPI value (from real application state, or 'N/A')
    - Contextual subtitle / status indicator
    """
    def __init__(self, parent, title: str, initial_value: str = "N/A", subtitle: str = "",
                 accent_color: str = COLOR_ACCENT, **kwargs):
        super().__init__(parent, style="MetricCard.TFrame", **kwargs)
        self.accent_color = accent_color

        # Top Category Label
        self.title_label = tk.Label(
            self, text=title.upper(), font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL, anchor="w"
        )
        self.title_label.pack(fill=tk.X, padx=10, pady=(8, 2))

        # Big Value Label
        self.value_var = tk.StringVar(value=initial_value)
        self.value_label = tk.Label(
            self, textvariable=self.value_var, font=("Segoe UI", 18, "bold"),
            fg=accent_color, bg=COLOR_PANEL, anchor="w"
        )
        self.value_label.pack(fill=tk.X, padx=10, pady=(0, 2))

        # Subtitle / Hint
        self.sub_var = tk.StringVar(value=subtitle)
        self.sub_label = tk.Label(
            self, textvariable=self.sub_var, font=("Segoe UI", 8),
            fg=COLOR_TEXT_DIM, bg=COLOR_PANEL, anchor="w"
        )
        self.sub_label.pack(fill=tk.X, padx=10, pady=(0, 8))

    def set_value(self, value: str, color: Optional[str] = None, subtitle: Optional[str] = None):
        """Update card value, text color, and subtitle safely."""
        self.value_var.set(str(value) if value is not None and value != "" else "N/A")
        if color:
            self.value_label.configure(fg=color)
        if subtitle is not None:
            self.sub_var.set(subtitle)


class ApplicationGUI:
    """
    Modern AI Surveillance Dashboard GUI for Live Activity Analyzer.
    
    Provides:
    - Top bar with real-time status pill, FPS badge, clock, and source indicator
    - Left multi-page navigation sidebar (10 dedicated views)
    - Central container hosting modular view pages
    - Real-time metric cards with authentic pipeline values
    - Live Video Monitor with BGR->RGB color correction and aspect scaling
    - Camera fallback controls and reconnection indicator
    - Real-time active tracking and surveillance event tables
    - Activity mode selector with future-upgrade notices
    - Configuration management view
    - Academic context & architecture About view
    """

    def __init__(self, config_manager: ConfigManager, start_loops: bool = True):
        self.config_manager = config_manager
        self.config: AppConfig = config_manager.get_config()

        self.root = tk.Tk()
        self.root.title("Live Activity Analyzer — AI CCTV Surveillance Dashboard")
        self.root.geometry("1440x900")
        self.root.minsize(1200, 780)

        # Operational State
        self.running = False
        self.paused = False
        self.current_frame = None
        self.photo_image = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.current_activity_mode = "Auto Detect"

        # Camera & Hardware State
        self.camera_state = CameraState.DISCONNECTED
        self.camera_status_message = "Ready"
        self.camera_devices: List[CameraDeviceInfo] = []
        self.selected_camera_id: Optional[int] = None
        self.source_description: str = "No Source"

        # Status and Telemetry string variables (backward compatible with original API)
        self.status_vars: Dict[str, tk.StringVar] = {
            "camera": tk.StringVar(value="Disconnected"),
            "fps": tk.StringVar(value="0.0"),
            "people_count": tk.StringVar(value="0"),
            "current_activity": tk.StringVar(value="None"),
            "anomaly_status": tk.StringVar(value="NORMAL"),
            "alert_status": tk.StringVar(value="No Alerts"),
            "active_tracks": tk.StringVar(value="0"),
            "objects_count": tk.StringVar(value="0"),
            "inference_time": tk.StringVar(value="N/A"),
            "process_time": tk.StringVar(value="N/A"),
            "resolution": tk.StringVar(value="N/A"),
            "dropped_frames": tk.StringVar(value="0"),
            "pipeline_detection": tk.StringVar(value="IDLE"),
            "pipeline_tracking": tk.StringVar(value="IDLE"),
            "pipeline_activity": tk.StringVar(value="IDLE"),
            "pipeline_anomaly": tk.StringVar(value="IDLE"),
        }

        # Running statistics string variables (backward compatible)
        self.stats_vars: Dict[str, tk.StringVar] = {
            "total_detected": tk.StringVar(value="0"),
            "total_entries": tk.StringVar(value="0"),
            "total_exits": tk.StringVar(value="0"),
            "walking_count": tk.StringVar(value="0"),
            "standing_count": tk.StringVar(value="0"),
            "running_count": tk.StringVar(value="0"),
            "loitering_events": tk.StringVar(value="0"),
            "anomaly_count": tk.StringVar(value="0"),
            "restricted_zone_events": tk.StringVar(value="0"),
        }

        # Callback hooks (wired by wire_application in main.py)
        self.on_auto_start_camera: Optional[Callable] = None
        self.on_start_camera: Optional[Callable] = None
        self.on_stop_camera: Optional[Callable] = None
        self.on_pause: Optional[Callable] = None
        self.on_resume: Optional[Callable] = None
        self.on_load_video: Optional[Callable] = None
        self.on_connect_ip_camera: Optional[Callable] = None
        self.on_screenshot: Optional[Callable] = None
        self.on_export_report: Optional[Callable] = None
        self.on_clear_events: Optional[Callable] = None
        self.on_settings: Optional[Callable] = None
        self.on_retry_camera: Optional[Callable] = None
        self.on_select_camera: Optional[Callable[[int], None]] = None

        # Navigation registry and views container
        self.current_view_name = "live_monitor"
        self.sidebar_buttons: Dict[str, tk.Button] = {}
        self.views: Dict[str, ttk.Frame] = {}

        # Backward compatibility control buttons dict
        self.control_buttons: Dict[str, Any] = {}

        self._setup_styles()
        self._create_layout()
        if start_loops:
            self._start_gui_loops()

        logger.info("ApplicationGUI initialized successfully")

    # -----------------------------------------------------------------------
    # Theme & Styles
    # -----------------------------------------------------------------------
    def _setup_styles(self) -> None:
        """Configure ttk styles with a modern dark surveillance palette."""
        style = ttk.Style()
        style.theme_use("clam")

        self.root.configure(bg=COLOR_BG_DARK)

        style.configure(".", background=COLOR_BG_DARK, foreground=COLOR_TEXT_WHITE)
        style.configure("TFrame", background=COLOR_BG_DARK)
        style.configure("Sidebar.TFrame", background=COLOR_SIDEBAR)
        style.configure("TopBar.TFrame", background=COLOR_TOPBAR)
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        style.configure("MetricCard.TFrame", background=COLOR_PANEL, relief="solid", borderwidth=1)

        style.configure("TLabel", background=COLOR_BG_DARK, foreground=COLOR_TEXT_WHITE)
        style.configure("Panel.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT_WHITE)
        style.configure("Muted.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT_MUTED)

        # Buttons
        style.configure("TButton", background=COLOR_PANEL, foreground=COLOR_TEXT_WHITE,
                        borderwidth=1, focuscolor="none", font=("Segoe UI", 9, "bold"))
        style.map("TButton",
            background=[("active", COLOR_PANEL_HOVER), ("pressed", COLOR_ACCENT_HOVER)],
            foreground=[("active", COLOR_TEXT_WHITE), ("pressed", COLOR_TEXT_WHITE)]
        )

        style.configure("Accent.TButton", background=COLOR_ACCENT_HOVER, foreground=COLOR_TEXT_WHITE,
                        borderwidth=0, font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton",
            background=[("active", COLOR_ACCENT), ("pressed", "#03045e")]
        )

        style.configure("Danger.TButton", background="#7f1d1d", foreground=COLOR_TEXT_WHITE,
                        borderwidth=0, font=("Segoe UI", 9, "bold"))
        style.map("Danger.TButton",
            background=[("active", COLOR_DANGER), ("pressed", "#450a0a")]
        )

        # Combobox & Entry
        style.configure("TCombobox", fieldbackground=COLOR_PANEL, background=COLOR_PANEL,
                        foreground=COLOR_TEXT_WHITE, arrowcolor=COLOR_TEXT_WHITE)
        style.map("TCombobox", fieldbackground=[("readonly", COLOR_PANEL)])
        style.configure("TEntry", fieldbackground=COLOR_PANEL, foreground=COLOR_TEXT_WHITE)

        # Treeview (Surveillance Event & Tracks tables)
        style.configure("Treeview", background=COLOR_PANEL, foreground=COLOR_TEXT_WHITE,
                        fieldbackground=COLOR_PANEL, rowheight=24, borderwidth=0)
        style.configure("Treeview.Heading", background=COLOR_TOPBAR, foreground=COLOR_ACCENT,
                        font=("Segoe UI", 9, "bold"), borderwidth=1)
        style.map("Treeview", background=[("selected", COLOR_ACCENT_HOVER)],
                  foreground=[("selected", COLOR_TEXT_WHITE)])

    # -----------------------------------------------------------------------
    # Layout Shell: Top Bar, Sidebar, Main Stack, Bottom Bar
    # -----------------------------------------------------------------------
    def _create_layout(self) -> None:
        """Build the master UI shell."""
        # 1. Top Bar
        self._create_top_bar()

        # 2. Main Middle Paned Area (Sidebar + Content Stack)
        self.middle_frame = tk.Frame(self.root, bg=COLOR_BG_DARK)
        self.middle_frame.pack(fill=tk.BOTH, expand=True)

        # Left Sidebar
        self._create_sidebar(self.middle_frame)

        # Right Content Area (View Switcher Container)
        self.content_container = tk.Frame(self.middle_frame, bg=COLOR_BG_DARK)
        self.content_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Initialize All 10 Navigation Views
        self._create_views()

        # 3. Bottom Status Bar
        self._create_bottom_bar()

        # Default active view: Live Monitor
        self.show_view("live_monitor")

    # -----------------------------------------------------------------------
    # Top Telemetry & Header Bar
    # -----------------------------------------------------------------------
    def _create_top_bar(self) -> None:
        """Create the top surveillance header and telemetry bar."""
        self.top_bar = tk.Frame(self.root, bg=COLOR_TOPBAR, height=52)
        self.top_bar.pack(fill=tk.X, side=tk.TOP)
        self.top_bar.pack_propagate(False)

        # Left Brand Section
        brand_frame = tk.Frame(self.top_bar, bg=COLOR_TOPBAR)
        brand_frame.pack(side=tk.LEFT, padx=16, pady=6)

        title_lbl = tk.Label(
            brand_frame, text="LIVE ACTIVITY ANALYZER",
            font=("Segoe UI", 12, "bold"), fg=COLOR_TEXT_WHITE, bg=COLOR_TOPBAR
        )
        title_lbl.pack(side=tk.LEFT)

        badge_lbl = tk.Label(
            brand_frame, text="AI SURVEILLANCE v1.0",
            font=("Segoe UI", 8, "bold"), fg=COLOR_ACCENT, bg=COLOR_PANEL, padx=6, pady=2
        )
        badge_lbl.pack(side=tk.LEFT, padx=10)

        # Right Telemetry Section
        telemetry_frame = tk.Frame(self.top_bar, bg=COLOR_TOPBAR)
        telemetry_frame.pack(side=tk.RIGHT, padx=16, pady=6)

        # System Status Pill
        self.top_status_pill = tk.Label(
            telemetry_frame, text="● SYSTEM READY", font=("Segoe UI", 9, "bold"),
            fg=COLOR_SUCCESS, bg=COLOR_PANEL, padx=10, pady=4
        )
        self.top_status_pill.pack(side=tk.LEFT, padx=6)

        # FPS Badge
        self.top_fps_badge = tk.Label(
            telemetry_frame, text="FPS: 0.0", font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL, padx=10, pady=4
        )
        self.top_fps_badge.pack(side=tk.LEFT, padx=6)

        # Source Badge
        self.top_source_badge = tk.Label(
            telemetry_frame, text="SOURCE: NONE", font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL, padx=10, pady=4
        )
        self.top_source_badge.pack(side=tk.LEFT, padx=6)

        # Clock
        self.top_clock_label = tk.Label(
            telemetry_frame, text="--:--:--", font=("Consolas", 10),
            fg=COLOR_TEXT_MUTED, bg=COLOR_TOPBAR, padx=8
        )
        self.top_clock_label.pack(side=tk.LEFT, padx=4)

    # -----------------------------------------------------------------------
    # Left Navigation Sidebar
    # -----------------------------------------------------------------------
    def _create_sidebar(self, parent: tk.Frame) -> None:
        """Create the modern sidebar with 10 actual view buttons."""
        self.sidebar_frame = tk.Frame(parent, bg=COLOR_SIDEBAR, width=200)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False)

        # Header in sidebar
        nav_title = tk.Label(
            self.sidebar_frame, text="NAVIGATION", font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_DIM, bg=COLOR_SIDEBAR, anchor="w"
        )
        nav_title.pack(fill=tk.X, padx=16, pady=(16, 8))

        nav_items = [
            ("dashboard", "⚡ Dashboard"),
            ("live_monitor", "📹 Live Monitor"),
            ("people", "👥 People"),
            ("activities", "🏃 Activities"),
            ("anomalies", "⚠️ Anomalies"),
            ("statistics", "📊 Statistics"),
            ("events", "📋 Events"),
            ("camera", "📷 Camera"),
            ("settings", "⚙ Settings"),
            ("about", "ℹ About"),
        ]

        for key, label in nav_items:
            btn = tk.Button(
                self.sidebar_frame, text=f"  {label}", font=("Segoe UI", 10),
                fg=COLOR_TEXT_MUTED, bg=COLOR_SIDEBAR, activebackground=COLOR_PANEL,
                activeforeground=COLOR_TEXT_WHITE, bd=0, relief="flat", anchor="w",
                padx=12, pady=8, cursor="hand2",
                command=lambda k=key: self.show_view(k)
            )
            btn.pack(fill=tk.X, padx=8, pady=2)
            self.sidebar_buttons[key] = btn

        # Quick Control Shortcut Section at Bottom of Sidebar
        quick_frame = tk.Frame(self.sidebar_frame, bg=COLOR_SIDEBAR)
        quick_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=12)

        sep = tk.Frame(quick_frame, bg=COLOR_BORDER, height=1)
        sep.pack(fill=tk.X, pady=(0, 10))

        quick_lbl = tk.Label(
            quick_frame, text="QUICK ACTIONS", font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_DIM, bg=COLOR_SIDEBAR, anchor="w"
        )
        quick_lbl.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.side_start_btn = tk.Button(
            quick_frame, text="▶ Start Camera", font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_WHITE, bg=COLOR_ACCENT_HOVER, activebackground=COLOR_ACCENT,
            bd=0, relief="flat", pady=6, cursor="hand2", command=self._on_start_click
        )
        self.side_start_btn.pack(fill=tk.X, pady=2)

        self.side_stop_btn = tk.Button(
            quick_frame, text="⏹ Stop Camera", font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_WHITE, bg="#7f1d1d", activebackground=COLOR_DANGER,
            bd=0, relief="flat", pady=6, cursor="hand2", command=self._on_stop_click,
            state=tk.DISABLED
        )
        self.side_stop_btn.pack(fill=tk.X, pady=2)

    # -----------------------------------------------------------------------
    # View Switching Architecture
    # -----------------------------------------------------------------------
    def show_view(self, view_name: str) -> None:
        """Switch the main container view to the requested view name."""
        if view_name not in self.views:
            logger.warning(f"Requested view '{view_name}' does not exist.")
            return

        self.current_view_name = view_name

        # Hide all views
        for v in self.views.values():
            v.pack_forget()

        # Display target view
        self.views[view_name].pack(fill=tk.BOTH, expand=True)

        # Update sidebar button visual styles
        for key, btn in self.sidebar_buttons.items():
            if key == view_name:
                btn.configure(bg=COLOR_PANEL, fg=COLOR_ACCENT, font=("Segoe UI", 10, "bold"))
            else:
                btn.configure(bg=COLOR_SIDEBAR, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 10))

        logger.debug(f"Switched view to '{view_name}'")

    # -----------------------------------------------------------------------
    # Create All 10 Navigation Views
    # -----------------------------------------------------------------------
    def _create_views(self) -> None:
        """Initialize the 10 distinct surveillance dashboard views."""
        self._create_dashboard_view()
        self._create_live_monitor_view()
        self._create_people_view()
        self._create_activities_view()
        self._create_anomalies_view()
        self._create_statistics_view()
        self._create_events_view()
        self._create_camera_view()
        self._create_settings_view()
        self._create_about_view()

    # -----------------------------------------------------------------------
    # View 1: Dashboard View (7 Metric Cards + Overview)
    # -----------------------------------------------------------------------
    def _create_dashboard_view(self) -> None:
        dash_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["dashboard"] = dash_frame

        # Top Section: 7 Metric Cards Grid
        cards_container = tk.Frame(dash_frame, bg=COLOR_BG_DARK)
        cards_container.pack(fill=tk.X, pady=(0, 10))

        for col in range(7):
            cards_container.columnconfigure(col, weight=1, uniform="metric_cards")

        self.metric_cards: Dict[str, MetricCard] = {}

        cards_spec = [
            ("people", "PEOPLE", "0", "Detected in frame", COLOR_ACCENT),
            ("active_tracks", "ACTIVE TRACKS", "0", "ByteTrack active", "#38bdf8"),
            ("objects", "OBJECTS", "0", "Total Detections", "#a78bfa"),
            ("fps", "FPS", "0.0", "Inference speed", COLOR_SUCCESS),
            ("camera_status", "CAMERA STATUS", "DISCONNECTED", "Hardware state", COLOR_WARNING),
            ("current_activity", "CURRENT ACTIVITY", "None", "Dominant motion", "#f472b6"),
            ("anomalies", "ANOMALIES", "NORMAL", "Isolation Forest", COLOR_SUCCESS),
        ]

        for col_idx, (key, title, val, sub, color) in enumerate(cards_spec):
            card = MetricCard(cards_container, title=title, initial_value=val, subtitle=sub, accent_color=color)
            card.grid(row=0, column=col_idx, padx=4, sticky="nsew")
            self.metric_cards[key] = card

        # Middle Split: System Telemetry & Activity Summary
        mid_container = tk.Frame(dash_frame, bg=COLOR_BG_DARK)
        mid_container.pack(fill=tk.BOTH, expand=True)

        # Left Card: System Overview & Quick Controls
        left_card = tk.Frame(mid_container, bg=COLOR_PANEL, relief="solid", borderwidth=1)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(
            left_card, text="SYSTEM OVERVIEW & CONTROLS", font=("Segoe UI", 10, "bold"),
            fg=COLOR_ACCENT, bg=COLOR_PANEL
        ).pack(anchor="w", padx=12, pady=(10, 6))

        sys_info_text = (
            "• Engine: YOLOv8 Person Detection + ByteTrack Association\n"
            "• Anomaly Detection: Scikit-Learn IsolationForest (Online Scoring)\n"
            "• Database: SQLite Session & Event Logging Active\n"
            "• Video Rendering: OpenCV BGR to RGB Hardware Calibrated"
        )
        tk.Label(
            left_card, text=sys_info_text, font=("Segoe UI", 9),
            fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL, justify="left"
        ).pack(anchor="w", padx=12, pady=4)

        # Quick action button row
        dash_btn_row = tk.Frame(left_card, bg=COLOR_PANEL)
        dash_btn_row.pack(anchor="w", padx=12, pady=10)

        b1 = ttk.Button(dash_btn_row, text="▶ Start Camera", command=self._on_start_click)
        b1.pack(side=tk.LEFT, padx=4)
        b2 = ttk.Button(dash_btn_row, text="⏹ Stop Camera", command=self._on_stop_click)
        b2.pack(side=tk.LEFT, padx=4)
        b3 = ttk.Button(dash_btn_row, text="📁 Load Video", command=self._on_load_video_click)
        b3.pack(side=tk.LEFT, padx=4)
        b4 = ttk.Button(dash_btn_row, text="🔄 Retry Camera", command=self._on_retry_click)
        b4.pack(side=tk.LEFT, padx=4)

        # Right Card: Activity Distribution Breakdown
        right_card = tk.Frame(mid_container, bg=COLOR_PANEL, relief="solid", borderwidth=1)
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        tk.Label(
            right_card, text="REAL-TIME ACTIVITY DISTRIBUTION", font=("Segoe UI", 10, "bold"),
            fg=COLOR_ACCENT, bg=COLOR_PANEL
        ).pack(anchor="w", padx=12, pady=(10, 6))

        self.dash_activity_labels = {}
        act_items = [
            ("Walking", "walking_count"),
            ("Standing", "standing_count"),
            ("Running", "running_count"),
            ("Loitering", "loitering_events"),
            ("Zone Events", "restricted_zone_events"),
        ]
        for name, key in act_items:
            row = tk.Frame(right_card, bg=COLOR_PANEL)
            row.pack(fill=tk.X, padx=12, pady=3)
            tk.Label(row, text=name, font=("Segoe UI", 9), fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL).pack(side=tk.LEFT)
            val_lbl = tk.Label(row, textvariable=self.stats_vars[key], font=("Segoe UI", 9, "bold"),
                               fg=COLOR_ACCENT, bg=COLOR_PANEL)
            val_lbl.pack(side=tk.RIGHT)
            self.dash_activity_labels[name] = val_lbl

        # Bottom Stream: Recent Events Preview
        bot_card = tk.Frame(dash_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, height=140)
        bot_card.pack(fill=tk.X, pady=(10, 0))
        bot_card.pack_propagate(False)

        tk.Label(
            bot_card, text="LIVE SURVEILLANCE EVENT STREAM", font=("Segoe UI", 9, "bold"),
            fg=COLOR_ACCENT, bg=COLOR_PANEL
        ).pack(anchor="w", padx=12, pady=(6, 2))

        self.dash_recent_event_lbl = tk.Label(
            bot_card, text="No events recorded yet. Pipeline active.",
            font=("Consolas", 9), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL, anchor="w"
        )
        self.dash_recent_event_lbl.pack(fill=tk.X, padx=12, pady=4)

    # -----------------------------------------------------------------------
    # View 2: Live Monitor View (Prominent Video Canvas + Mode Selector)
    # -----------------------------------------------------------------------
    def _create_live_monitor_view(self) -> None:
        monitor_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["live_monitor"] = monitor_frame

        # 1. Top Live Camera Info & Telemetry Bar
        header_bar = tk.Frame(monitor_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, height=36)
        header_bar.pack(fill=tk.X, pady=(0, 4))
        header_bar.pack_propagate(False)

        # Left: Live Camera Source Title
        self.live_camera_title_lbl = tk.Label(
            header_bar, text="LIVE CAMERA: No Source", font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL
        )
        self.live_camera_title_lbl.pack(side=tk.LEFT, padx=10)

        # Middle: Subsystem Status Indicators (Detection, Tracking, Activity, Anomaly)
        subsys_frame = tk.Frame(header_bar, bg=COLOR_PANEL)
        subsys_frame.pack(side=tk.LEFT, padx=10)

        self.badge_det = tk.Label(subsys_frame, text="DET: IDLE", font=("Segoe UI", 8, "bold"),
                                  fg=COLOR_TEXT_DIM, bg=COLOR_TOPBAR, padx=6, pady=2)
        self.badge_det.pack(side=tk.LEFT, padx=2)

        self.badge_trk = tk.Label(subsys_frame, text="TRK: IDLE", font=("Segoe UI", 8, "bold"),
                                  fg=COLOR_TEXT_DIM, bg=COLOR_TOPBAR, padx=6, pady=2)
        self.badge_trk.pack(side=tk.LEFT, padx=2)

        self.badge_act = tk.Label(subsys_frame, text="ACT: IDLE", font=("Segoe UI", 8, "bold"),
                                  fg=COLOR_TEXT_DIM, bg=COLOR_TOPBAR, padx=6, pady=2)
        self.badge_act.pack(side=tk.LEFT, padx=2)

        self.badge_anom = tk.Label(subsys_frame, text="ANOM: IDLE", font=("Segoe UI", 8, "bold"),
                                   fg=COLOR_TEXT_DIM, bg=COLOR_TOPBAR, padx=6, pady=2)
        self.badge_anom.pack(side=tk.LEFT, padx=2)

        # Right: Quick Header Telemetry (Resolution, FPS, Inference)
        self.live_telemetry_top_lbl = tk.Label(
            header_bar, text="RES: N/A | FPS: 0.0 | INFER: N/A", font=("Consolas", 8, "bold"),
            fg=COLOR_ACCENT, bg=COLOR_PANEL
        )
        self.live_telemetry_top_lbl.pack(side=tk.RIGHT, padx=10)

        # 2. Activity Mode Selector Bar
        mode_bar = tk.Frame(monitor_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, height=40)
        mode_bar.pack(fill=tk.X, pady=(0, 4))
        mode_bar.pack_propagate(False)

        mode_title = tk.Label(
            mode_bar, text="ACTIVITY MODE:", font=("Segoe UI", 8, "bold"),
            fg=COLOR_ACCENT, bg=COLOR_PANEL
        )
        mode_title.pack(side=tk.LEFT, padx=10)

        self.mode_buttons: Dict[str, tk.Button] = {}
        # Posture mode is enabled when pose estimator is available
        pose_available = hasattr(self, "_pose_available") and self._pose_available
        gesture_available = hasattr(self, "_gesture_available") and self._gesture_available
        face_available = hasattr(self, "_face_available") and self._face_available
        modes = [
            ("Auto Detect", True, ""),
            ("Body Activity", True, ""),
            ("Movement", True, ""),
            ("Posture", pose_available, "Upcoming: UPGRADE 7" if not pose_available else ""),
            ("Hand Gesture", gesture_available, "Upcoming: UPGRADE 8" if not gesture_available else ""),
            ("Face", face_available, "Upcoming: UPGRADE 9" if not face_available else ""),
        ]

        for mode_name, is_implemented, badge in modes:
            btn_text = mode_name if not badge else f"{mode_name} ({badge})"
            btn_state = tk.NORMAL if is_implemented else tk.DISABLED
            btn = tk.Button(
                mode_bar, text=btn_text, font=("Segoe UI", 8, "bold" if is_implemented else "normal"),
                fg=COLOR_TEXT_WHITE if is_implemented else COLOR_TEXT_DIM,
                bg=COLOR_ACCENT_HOVER if mode_name == "Auto Detect" else COLOR_PANEL,
                activebackground=COLOR_ACCENT, bd=0, relief="flat", padx=6, pady=3, cursor="hand2",
                state=btn_state,
                command=lambda m=mode_name, imp=is_implemented: self._on_activity_mode_selected(m, imp)
            )
            btn.pack(side=tk.LEFT, padx=3, pady=5)
            self.mode_buttons[mode_name] = btn

        # 3. Center Area Split: Video Canvas (Left) + Performance & Event Stream Panel (Right)
        center_split = tk.Frame(monitor_frame, bg=COLOR_BG_DARK)
        center_split.pack(fill=tk.BOTH, expand=True)

        # Left: Video Wrapper
        video_wrapper = tk.Frame(center_split, bg="#0a0c10", relief="solid", borderwidth=1)
        video_wrapper.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(
            video_wrapper, bg="#0a0c10", highlightthickness=0
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Fallback Toolbar (Visible when camera is disconnected / searching)
        self.fallback_frame = tk.Frame(video_wrapper, bg=COLOR_PANEL, padx=4, pady=4)
        self.fallback_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.retry_cam_btn = ttk.Button(self.fallback_frame, text="🔄 Retry Camera", command=self._on_retry_click)
        self.retry_cam_btn.pack(side=tk.LEFT, padx=4)

        self.select_cam_btn = ttk.Button(self.fallback_frame, text="📷 Select Camera", command=self._on_select_camera_dialog)
        self.select_cam_btn.pack(side=tk.LEFT, padx=4)

        self.use_file_btn = ttk.Button(self.fallback_frame, text="📁 Use Video File", command=self._on_load_video_click)
        self.use_file_btn.pack(side=tk.LEFT, padx=4)

        self.cam_settings_btn = ttk.Button(self.fallback_frame, text="⚙ Camera Settings", command=self._on_settings_click)
        self.cam_settings_btn.pack(side=tk.LEFT, padx=4)

        self.video_info_var = tk.StringVar(value="Camera system ready.")
        self.video_info_label = tk.Label(
            self.fallback_frame, textvariable=self.video_info_var,
            font=("Consolas", 9), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL
        )
        self.video_info_label.pack(side=tk.RIGHT, padx=8)

        # Right: Live Operator Panel (Performance Monitor + Live Event Stream)
        side_panel = tk.Frame(center_split, bg=COLOR_BG_DARK, width=320)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        side_panel.pack_propagate(False)

        # Card 1: Performance Monitor
        perf_card = tk.Frame(side_panel, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=10, pady=8)
        perf_card.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            perf_card, text="PERFORMANCE & TELEMETRY", font=("Segoe UI", 9, "bold"),
            fg=COLOR_ACCENT, bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 6))

        perf_grid = tk.Frame(perf_card, bg=COLOR_PANEL)
        perf_grid.pack(fill=tk.X)

        perf_rows = [
            ("Resolution:", "resolution", COLOR_TEXT_WHITE),
            ("Inference FPS:", "fps", COLOR_SUCCESS),
            ("YOLO Inference:", "inference_time", COLOR_ACCENT),
            ("Total Latency:", "process_time", COLOR_WARNING),
            ("Active Tracks:", "active_tracks", "#38bdf8"),
            ("Objects Count:", "objects_count", "#a78bfa"),
            ("Dropped Frames:", "dropped_frames", COLOR_TEXT_MUTED),
        ]

        self.perf_labels = {}
        for r_idx, (lbl_text, var_key, val_color) in enumerate(perf_rows):
            tk.Label(perf_grid, text=lbl_text, font=("Segoe UI", 8), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL).grid(
                row=r_idx, column=0, sticky="w", pady=2
            )
            val_lbl = tk.Label(
                perf_grid, textvariable=self.status_vars[var_key], font=("Segoe UI", 8, "bold"),
                fg=val_color, bg=COLOR_PANEL
            )
            val_lbl.grid(row=r_idx, column=1, sticky="e", padx=(8, 0), pady=2)
            perf_grid.columnconfigure(1, weight=1)
            self.perf_labels[var_key] = val_lbl

        # Card 2: Live Incident & Event Stream
        event_card = tk.Frame(side_panel, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=8, pady=8)
        event_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            event_card, text="LIVE EVENT STREAM", font=("Segoe UI", 9, "bold"),
            fg=COLOR_ACCENT, bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 4))

        evt_cols = ("Time", "ID", "Event", "Details")
        self.live_events_tree = ttk.Treeview(event_card, columns=evt_cols, show="headings", height=8)

        self.live_events_tree.heading("Time", text="Time")
        self.live_events_tree.column("Time", width=62, anchor=tk.CENTER)

        self.live_events_tree.heading("ID", text="ID")
        self.live_events_tree.column("ID", width=48, anchor=tk.CENTER)

        self.live_events_tree.heading("Event", text="Event")
        self.live_events_tree.column("Event", width=80, anchor=tk.W)

        self.live_events_tree.heading("Details", text="Details")
        self.live_events_tree.column("Details", width=105, anchor=tk.W)

        evt_scroll = ttk.Scrollbar(event_card, orient=tk.VERTICAL, command=self.live_events_tree.yview)
        self.live_events_tree.configure(yscrollcommand=evt_scroll.set)

        self.live_events_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        evt_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 4. Bottom Video Controls Bar
        controls_bar = tk.Frame(monitor_frame, bg=COLOR_PANEL, height=48, relief="solid", borderwidth=1)
        controls_bar.pack(fill=tk.X, pady=(6, 0))
        controls_bar.pack_propagate(False)

        buttons = [
            ("Start Camera", self._on_start_click, "start_btn"),
            ("Stop Camera", self._on_stop_click, "stop_btn"),
            ("Pause", self._on_pause_click, "pause_btn"),
            ("Resume", self._on_resume_click, "resume_btn"),
            ("Load Video", self._on_load_video_click, "load_btn"),
            ("Screenshot", self._on_screenshot_click, "screenshot_btn"),
            ("Export Report", self._on_export_report_click, "export_btn"),
        ]

        for text, cmd, key in buttons:
            btn = ttk.Button(controls_bar, text=text, command=cmd)
            btn.pack(side=tk.LEFT, padx=4, pady=8)
            self.control_buttons[key] = btn

        # Populate legacy keys
        self.control_buttons["ip_btn"] = ttk.Button(controls_bar, text="Connect IP", command=self._on_connect_ip_click)
        self.control_buttons["clear_btn"] = ttk.Button(controls_bar, text="Clear Events", command=self._on_clear_events_click)
        self.control_buttons["settings_btn"] = ttk.Button(controls_bar, text="Settings", command=self._on_settings_click)

        # Right Telemetry Pill in Video Control Bar
        self.video_telemetry_var = tk.StringVar(value="FPS: 0.0 | Infer: N/A | People: 0 | Tracks: 0 | State: DISCONNECTED")
        tk.Label(
            controls_bar, textvariable=self.video_telemetry_var,
            font=("Consolas", 9, "bold"), fg=COLOR_ACCENT, bg=COLOR_PANEL
        ).pack(side=tk.RIGHT, padx=12)

    def _on_activity_mode_selected(self, mode_name: str, is_implemented: bool) -> None:
        """Handle user changing activity detection mode."""
        if not is_implemented:
            upgrade_map = {
                "Posture": "UPGRADE 7 (Pose & Posture Recognition with YOLOv8-pose)",
                "Hand Gesture": "UPGRADE 8 (Hand Gesture Recognition)",
                "Face Gesture": "UPGRADE 9 (Face & Expression Detection)",
            }
            milestone = upgrade_map.get(mode_name, "a future milestone")
            messagebox.showinfo(
                "Feature Notice",
                f"'{mode_name}' is scheduled for {milestone}.\n\n"
                "Current engine will continue running kinematic body activity detection."
            )
            return

        self.current_activity_mode = mode_name
        for name, btn in self.mode_buttons.items():
            if name == mode_name:
                btn.configure(bg=COLOR_ACCENT_HOVER, fg=COLOR_TEXT_WHITE)
            elif name in ("Auto Detect", "Body Activity", "Movement"):
                btn.configure(bg=COLOR_PANEL, fg=COLOR_TEXT_WHITE)

        self.video_telemetry_var.set(
            f"Mode: {mode_name} | Tracks: {self.status_vars['active_tracks'].get()} | Anomaly: {self.status_vars['anomaly_status'].get()}"
        )
        self.set_status_bar(f"Activity mode switched to: {mode_name}")

    # -----------------------------------------------------------------------
    # View 3: People View (Live Track Table)
    # -----------------------------------------------------------------------
    def _create_people_view(self) -> None:
        people_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["people"] = people_frame

        header = tk.Frame(people_frame, bg=COLOR_BG_DARK)
        header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            header, text="ACTIVE PERSON TRACKING (BYTETRACK ENGINE)",
            font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT, bg=COLOR_BG_DARK
        ).pack(side=tk.LEFT)

        self.people_summary_lbl = tk.Label(
            header, text="Active Persons: 0 | Total Unique Detected: 0",
            font=("Segoe UI", 9), fg=COLOR_TEXT_MUTED, bg=COLOR_BG_DARK
        )
        self.people_summary_lbl.pack(side=tk.RIGHT)

        columns = ("Track ID", "Activity", "Posture", "Gesture", "Face", "Speed (px/s)", "Distance (px)", "Duration (s)", "Bounding Box (X1, Y1, X2, Y2)", "Status")
        self.people_tree = ttk.Treeview(people_frame, columns=columns, show="headings", height=18)

        for col in columns:
            self.people_tree.heading(col, text=col)
            self.people_tree.column(col, width=120, anchor=tk.CENTER)

        self.people_tree.column("Track ID", width=90)
        self.people_tree.column("Posture", width=110)
        self.people_tree.column("Gesture", width=130)
        self.people_tree.column("Face", width=80)
        self.people_tree.column("Bounding Box (X1, Y1, X2, Y2)", width=240)

        scrollbar = ttk.Scrollbar(people_frame, orient=tk.VERTICAL, command=self.people_tree.yview)
        self.people_tree.configure(yscrollcommand=scrollbar.set)

        self.people_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # -----------------------------------------------------------------------
    # View 4: Activities View (Breakdown + Mode Status)
    # -----------------------------------------------------------------------
    def _create_activities_view(self) -> None:
        act_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["activities"] = act_frame

        tk.Label(
            act_frame, text="ACTIVITY CLASSIFICATION & BEHAVIORAL ANALYTICS",
            font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT, bg=COLOR_BG_DARK
        ).pack(anchor="w", pady=(0, 12))

        # Cards row
        kpi_row = tk.Frame(act_frame, bg=COLOR_BG_DARK)
        kpi_row.pack(fill=tk.X, pady=(0, 16))

        for col in range(5):
            kpi_row.columnconfigure(col, weight=1, uniform="act_kpi")

        self.act_cards = {}
        act_specs = [
            ("walking", "WALKING", "0", "Speed 1.5 - 4.0 px/s", COLOR_SUCCESS),
            ("standing", "STANDING", "0", "Speed < 1.5 px/s", "#38bdf8"),
            ("running", "RUNNING", "0", "Speed >= 4.0 px/s", COLOR_WARNING),
            ("loitering", "LOITERING", "0", "Stationary in zone", COLOR_DANGER),
            ("zones", "RESTRICTED ZONES", "0", "Polygon breach events", COLOR_DANGER),
        ]
        for col, (k, title, val, sub, color) in enumerate(act_specs):
            card = MetricCard(kpi_row, title=title, initial_value=val, subtitle=sub, accent_color=color)
            card.grid(row=0, column=col, padx=4, sticky="nsew")
            self.act_cards[k] = card

        # Info Box explaining active and upcoming models
        info_card = tk.Frame(act_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=16, pady=16)
        info_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            info_card, text="CURRENT ACTIVITY CLASSIFICATION PIPELINE",
            font=("Segoe UI", 10, "bold"), fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 8))

        txt = (
            "• Kinematic Velocity Classification: Active (Standing, Walking, Running via centroid displacement)\n"
            "• Spatial Zone Violations: Active (Polygon ray-casting check against user-configured restricted zones)\n"
            "• Loitering Detection: Active (Positional variance < 100 over configurable duration threshold)\n\n"
            "UPCOMING ACTIVITY EXTENSIONS:\n"
            "• UPGRADE 7: YOLOv8-pose 17-keypoint skeletal posture estimation (sitting, lying, bending, fallen)\n"
            "• UPGRADE 8: Hand gesture recognition (pointing, raised hand, waving)\n"
            "• UPGRADE 9: Facial expression and crowd attention analytics"
        )
        tk.Label(
            info_card, text=txt, font=("Segoe UI", 9), fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL, justify="left"
        ).pack(anchor="w")

    # -----------------------------------------------------------------------
    # View 5: Anomalies View (Isolation Forest Model & Events)
    # -----------------------------------------------------------------------
    def _create_anomalies_view(self) -> None:
        anom_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["anomalies"] = anom_frame

        tk.Label(
            anom_frame, text="AI BEHAVIORAL ANOMALY DETECTION (ISOLATION FOREST)",
            font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT, bg=COLOR_BG_DARK
        ).pack(anchor="w", pady=(0, 10))

        # Model Info Header
        model_card = tk.Frame(anom_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=12, pady=10)
        model_card.pack(fill=tk.X, pady=(0, 10))

        m_text = (
            "Model: Scikit-learn IsolationForest  |  Contamination: 0.10  |  Estimators: 100  |  "
            "Features: [speed, distance, direction_changes, zone_time, count, transitions]"
        )
        tk.Label(
            model_card, text=m_text, font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL
        ).pack(anchor="w")

        note_text = "Explainable Anomaly Reasons (Feature attribution) is scheduled for UPGRADE 13."
        tk.Label(
            model_card, text=note_text, font=("Segoe UI", 8),
            fg=COLOR_TEXT_DIM, bg=COLOR_PANEL
        ).pack(anchor="w", pady=(4, 0))

        # Anomaly Log Table
        columns = ("Timestamp", "Person Track ID", "Anomaly Score", "Status", "Details")
        self.anomaly_tree = ttk.Treeview(anom_frame, columns=columns, show="headings", height=16)

        for col in columns:
            self.anomaly_tree.heading(col, text=col)
            self.anomaly_tree.column(col, width=140, anchor=tk.CENTER)

        self.anomaly_tree.column("Details", width=300, anchor=tk.W)

        scrollbar = ttk.Scrollbar(anom_frame, orient=tk.VERTICAL, command=self.anomaly_tree.yview)
        self.anomaly_tree.configure(yscrollcommand=scrollbar.set)

        self.anomaly_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # -----------------------------------------------------------------------
    # View 6: Statistics View (Pandas KPIs + Reports)
    # -----------------------------------------------------------------------
    def _create_statistics_view(self) -> None:
        stats_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["statistics"] = stats_frame

        tk.Label(
            stats_frame, text="SURVEILLANCE TELEMETRY & HISTORICAL STATISTICS",
            font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT, bg=COLOR_BG_DARK
        ).pack(anchor="w", pady=(0, 12))

        # Stats Cards
        grid = tk.Frame(stats_frame, bg=COLOR_BG_DARK)
        grid.pack(fill=tk.X, pady=(0, 16))

        for col in range(4):
            grid.columnconfigure(col, weight=1, uniform="stats_grid")

        specs = [
            ("total_detected", "TOTAL DETECTIONS", "0", "Cumulative persons seen"),
            ("total_entries", "ENTRIES", "0", "Zone boundary entries"),
            ("total_exits", "EXITS", "0", "Zone boundary exits"),
            ("anomaly_count", "TOTAL ANOMALIES", "0", "Flagged by ML engine"),
        ]

        self.stat_cards = {}
        for col, (key, title, val, sub) in enumerate(specs):
            card = MetricCard(grid, title=title, initial_value=val, subtitle=sub, accent_color=COLOR_ACCENT)
            card.grid(row=0, column=col, padx=4, sticky="nsew")
            self.stat_cards[key] = card

        # Action Box
        action_card = tk.Frame(stats_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=16, pady=16)
        action_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            action_card, text="AUTOMATED REPORT GENERATION",
            font=("Segoe UI", 10, "bold"), fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            action_card,
            text="Generate structured summary reports including time-series distributions, "
                 "person activity summaries, and incident records.",
            font=("Segoe UI", 9), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 12))

        b = ttk.Button(action_card, text="📥 Export Session Report (CSV / JSON)", command=self._on_export_report_click)
        b.pack(anchor="w")

    # -----------------------------------------------------------------------
    # View 7: Events View (Full Surveillance Log)
    # -----------------------------------------------------------------------
    def _create_events_view(self) -> None:
        events_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["events"] = events_frame

        header = tk.Frame(events_frame, bg=COLOR_BG_DARK)
        header.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            header, text="REAL-TIME SURVEILLANCE EVENT LOG",
            font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT, bg=COLOR_BG_DARK
        ).pack(side=tk.LEFT)

        btn_row = tk.Frame(header, bg=COLOR_BG_DARK)
        btn_row.pack(side=tk.RIGHT)

        ttk.Button(btn_row, text="🗑 Clear Events", command=self._on_clear_events_click).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="📥 Export Report", command=self._on_export_report_click).pack(side=tk.LEFT, padx=4)

        # Full Table
        columns = ("Time", "Type", "Person ID", "Activity", "Zone", "Details")
        self.events_tree = ttk.Treeview(events_frame, columns=columns, show="headings", height=20)

        for col in columns:
            self.events_tree.heading(col, text=col)
            self.events_tree.column(col, width=100, anchor=tk.W)

        self.events_tree.column("Time", width=100)
        self.events_tree.column("Details", width=350)

        scrollbar = ttk.Scrollbar(events_frame, orient=tk.VERTICAL, command=self.events_tree.yview)
        self.events_tree.configure(yscrollcommand=scrollbar.set)

        self.events_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # -----------------------------------------------------------------------
    # View 8: Camera Management View (Discovery & Stream Connectors)
    # -----------------------------------------------------------------------
    def _create_camera_view(self) -> None:
        cam_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["camera"] = cam_frame

        tk.Label(
            cam_frame, text="CAMERA HARDWARE DISCOVERY & STREAM CONNECTIVITY",
            font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT, bg=COLOR_BG_DARK
        ).pack(anchor="w", pady=(0, 12))

        # Panel 1: Local Device Selector
        p1 = tk.Frame(cam_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=16, pady=14)
        p1.pack(fill=tk.X, pady=(0, 10))

        tk.Label(p1, text="LOCAL HARDWARE CAMERAS", font=("Segoe UI", 10, "bold"),
                 fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL).pack(anchor="w", pady=(0, 6))

        sel_row = tk.Frame(p1, bg=COLOR_PANEL)
        sel_row.pack(fill=tk.X, pady=4)

        tk.Label(sel_row, text="Detected Devices:", font=("Segoe UI", 9),
                 fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL).pack(side=tk.LEFT, padx=(0, 10))

        self.camera_selector_var = tk.StringVar(value="Auto-detecting...")
        self.camera_combobox = ttk.Combobox(
            sel_row, textvariable=self.camera_selector_var, state="readonly", width=36
        )
        self.camera_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.camera_combobox.bind("<<ComboboxSelected>>", self._on_combobox_camera_selected)

        ttk.Button(sel_row, text="Connect Camera", command=self._on_start_selected_camera).pack(side=tk.LEFT, padx=4)
        ttk.Button(sel_row, text="🔄 Rescan Devices", command=self._on_retry_click).pack(side=tk.LEFT, padx=4)

        # Panel 2: RTSP / IP Camera Connector
        p2 = tk.Frame(cam_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=16, pady=14)
        p2.pack(fill=tk.X, pady=(0, 10))

        tk.Label(p2, text="IP CAMERA / RTSP STREAM", font=("Segoe UI", 10, "bold"),
                 fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL).pack(anchor="w", pady=(0, 6))

        ip_row = tk.Frame(p2, bg=COLOR_PANEL)
        ip_row.pack(fill=tk.X, pady=4)

        tk.Label(ip_row, text="Stream URL:", font=("Segoe UI", 9),
                 fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL).pack(side=tk.LEFT, padx=(0, 10))

        self.ip_url_var = tk.StringVar(value="rtsp://192.168.1.100:554/stream1")
        ip_entry = ttk.Entry(ip_row, textvariable=self.ip_url_var, width=42)
        ip_entry.pack(side=tk.LEFT, padx=(0, 10))

        def _connect_ip():
            url = self.ip_url_var.get().strip()
            if url and self.on_connect_ip_camera:
                self.set_status_bar(f"Connecting to stream: {url}...")
                self.on_connect_ip_camera(url)
                self.show_view("live_monitor")

        ttk.Button(ip_row, text="Connect Stream", command=_connect_ip).pack(side=tk.LEFT, padx=4)

        # Panel 3: Video File Playback
        p3 = tk.Frame(cam_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=16, pady=14)
        p3.pack(fill=tk.X, pady=(0, 10))

        tk.Label(p3, text="OFFLINE VIDEO FILE PLAYBACK", font=("Segoe UI", 10, "bold"),
                 fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL).pack(anchor="w", pady=(0, 6))

        tk.Label(
            p3, text="Select recorded CCTV video footage (.mp4, .avi, .mov) for automated offline surveillance analysis.",
            font=("Segoe UI", 9), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL
        ).pack(anchor="w", pady=(0, 8))

        ttk.Button(p3, text="📁 Browse & Load Video File", command=self._on_load_video_click).pack(anchor="w")

    def _on_start_selected_camera(self) -> None:
        """Start the camera candidate currently selected in the combobox."""
        if self.selected_camera_id is not None:
            if self.on_select_camera:
                self.on_select_camera(self.selected_camera_id)
            elif self.on_start_camera:
                self.on_start_camera(self.selected_camera_id)
            self.show_view("live_monitor")

    # -----------------------------------------------------------------------
    # View 9: Settings View (Pipeline & Configuration Form)
    # -----------------------------------------------------------------------
    def _create_settings_view(self) -> None:
        settings_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["settings"] = settings_frame

        tk.Label(
            settings_frame, text="SYSTEM & PIPELINE CONFIGURATION",
            font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT, bg=COLOR_BG_DARK
        ).pack(anchor="w", pady=(0, 12))

        form_card = tk.Frame(settings_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=20, pady=16)
        form_card.pack(fill=tk.X)

        fields = [
            ("Target Resolution (WxH):", "cam_res", f"{self.config.camera.width}x{self.config.camera.height}"),
            ("Target FPS:", "cam_fps", str(self.config.camera.fps)),
            ("YOLO Confidence Threshold:", "conf_thresh", str(self.config.detection.confidence_threshold)),
            ("YOLO IoU Threshold:", "iou_thresh", str(self.config.detection.iou_threshold)),
            ("Alert Cooldown Seconds:", "cooldown", str(self.config.alerts.cooldown_seconds)),
            ("Crowd Alert Threshold:", "crowd_thresh", str(self.config.alerts.crowd_threshold)),
        ]

        self.settings_entry_vars = {}
        for row_idx, (label_txt, var_key, default_val) in enumerate(fields):
            tk.Label(form_card, text=label_txt, font=("Segoe UI", 9, "bold"),
                     fg=COLOR_TEXT_WHITE, bg=COLOR_PANEL).grid(row=row_idx, column=0, sticky="w", pady=6)
            var = tk.StringVar(value=default_val)
            entry = ttk.Entry(form_card, textvariable=var, width=28)
            entry.grid(row=row_idx, column=1, sticky="w", padx=16, pady=6)
            self.settings_entry_vars[var_key] = var

        def _save_settings():
            try:
                res_str = self.settings_entry_vars["cam_res"].get().lower().strip()
                if "x" in res_str:
                    w, h = map(int, res_str.split("x"))
                    self.config_manager.set("camera.width", w)
                    self.config_manager.set("camera.height", h)

                self.config_manager.set("camera.fps", int(self.settings_entry_vars["cam_fps"].get()))
                self.config_manager.set("detection.confidence_threshold", float(self.settings_entry_vars["conf_thresh"].get()))
                self.config_manager.set("detection.iou_threshold", float(self.settings_entry_vars["iou_thresh"].get()))
                self.config_manager.set("alerts.cooldown_seconds", int(self.settings_entry_vars["cooldown"].get()))
                self.config_manager.set("alerts.crowd_threshold", int(self.settings_entry_vars["crowd_thresh"].get()))
                self.config_manager.save_config()

                self.set_status_bar("Configuration saved successfully.")
                messagebox.showinfo("Settings", "Configuration saved successfully.")
            except Exception as e:
                logger.error(f"Error saving settings: {e}")
                messagebox.showerror("Error", f"Failed to save settings: {e}")

        ttk.Button(form_card, text="💾 Save Configuration", command=_save_settings).grid(row=len(fields), column=1, sticky="w", padx=16, pady=14)

    # -----------------------------------------------------------------------
    # View 10: About View (Academic & Architectural Credits)
    # -----------------------------------------------------------------------
    def _create_about_view(self) -> None:
        about_frame = tk.Frame(self.content_container, bg=COLOR_BG_DARK)
        self.views["about"] = about_frame

        tk.Label(
            about_frame, text="ABOUT LIVE ACTIVITY ANALYZER",
            font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT, bg=COLOR_BG_DARK
        ).pack(anchor="w", pady=(0, 12))

        card = tk.Frame(about_frame, bg=COLOR_PANEL, relief="solid", borderwidth=1, padx=20, pady=16)
        card.pack(fill=tk.BOTH, expand=True)

        txt = (
            "LIVE ACTIVITY ANALYZER — v1.0.0\n"
            "AI-Based Real-Time CCTV Activity, Behavior and Anomaly Analysis System\n\n"
            "Academic Capstone Project (B.Tech Computer Science & Engineering, 5th Semester)\n"
            "Stack & Architecture:\n"
            "  • Object Detection: Ultralytics YOLOv8 (yolov8n.pt)\n"
            "  • Person Tracking: ByteTrack-style IoU Association & Trajectory Buffering\n"
            "  • Anomaly Engine: Scikit-Learn IsolationForest Feature Attribution\n"
            "  • Video Acquisition: OpenCV 5.0 Universal Hardware Discovery (AVFoundation/DirectShow/V4L2)\n"
            "  • User Interface: Python Tkinter/ttk Modern Dark Surveillance Dashboard\n"
            "  • Storage: SQLite Relational Database Engine + Pandas Reporting\n\n"
            "Milestone Status: UPGRADE 4 (Modern AI Surveillance Dashboard Redesign Complete)"
        )
        tk.Label(
            card, text=txt, font=("Segoe UI", 9), fg=COLOR_TEXT_WHITE,
            bg=COLOR_PANEL, justify="left"
        ).pack(anchor="w")

    # -----------------------------------------------------------------------
    # Bottom Status Bar
    # -----------------------------------------------------------------------
    def _create_bottom_bar(self) -> None:
        """Create the bottom status and persistent telemetry bar."""
        self.status_bar_frame = tk.Frame(self.root, bg=COLOR_TOPBAR, height=28)
        self.status_bar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_bar_frame.pack_propagate(False)

        self.status_bar_var = tk.StringVar(value="System Ready")
        tk.Label(
            self.status_bar_frame, textvariable=self.status_bar_var,
            font=("Segoe UI", 8), fg=COLOR_TEXT_MUTED, bg=COLOR_TOPBAR
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(
            self.status_bar_frame, text="Live Activity Analyzer v1.0.0  |  AI CCTV System",
            font=("Segoe UI", 8), fg=COLOR_TEXT_DIM, bg=COLOR_TOPBAR
        ).pack(side=tk.RIGHT, padx=10)

    # -----------------------------------------------------------------------
    # Video Canvas Rendering & Aspect Ratio Correction (BGR -> RGB)
    # -----------------------------------------------------------------------
    def _update_video_display(self) -> None:
        """Render current video frame with OpenCV BGR->RGB conversion and aspect scaling."""
        if self.current_frame is not None:
            try:
                canvas_w = self.video_canvas.winfo_width()
                canvas_h = self.video_canvas.winfo_height()

                if canvas_w > 1 and canvas_h > 1:
                    raw_frame = self.current_frame
                    h, w = raw_frame.shape[:2]

                    scale = min(canvas_w / w, canvas_h / h)
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

                    resized = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    # CRITICAL FIX: Convert BGR to RGB before Pillow rendering
                    if len(resized.shape) == 3 and resized.shape[2] == 3:
                        display_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    else:
                        display_rgb = resized

                    image = Image.fromarray(display_rgb)
                    self.photo_image = ImageTk.PhotoImage(image)

                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(
                        canvas_w // 2, canvas_h // 2,
                        image=self.photo_image, anchor=tk.CENTER
                    )
            except Exception as e:
                logger.debug(f"Video display rendering suppressed: {e}")
        else:
            self._render_canvas_placeholder()

    def _render_canvas_placeholder(self) -> None:
        """Render high-contrast, informative placeholder when video is inactive."""
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        self.video_canvas.delete("all")
        cx = canvas_w // 2
        cy = canvas_h // 2

        if self.camera_state == CameraState.SEARCHING:
            self.video_canvas.create_text(
                cx, cy - 25, text="🔍 Searching for Available Cameras...",
                fill="#00bcff", font=("Segoe UI", 16, "bold")
            )
            self.video_canvas.create_text(
                cx, cy + 15, text="Probing operating system hardware backends (AVFoundation / DirectShow / V4L2)...",
                fill=COLOR_TEXT_MUTED, font=("Segoe UI", 10)
            )
        elif self.camera_state == CameraState.CONNECTING:
            self.video_canvas.create_text(
                cx, cy - 25, text="🔄 Connecting to Video Device...",
                fill=COLOR_WARNING, font=("Segoe UI", 16, "bold")
            )
            self.video_canvas.create_text(
                cx, cy + 15, text=self.camera_status_message,
                fill=COLOR_TEXT_MUTED, font=("Segoe UI", 10)
            )
        elif self.camera_state == CameraState.RECONNECTING:
            self.video_canvas.create_text(
                cx, cy - 30, text="🔄 Camera Disconnected — Reconnecting...",
                fill=COLOR_ORANGE, font=("Segoe UI", 15, "bold")
            )
            self.video_canvas.create_text(
                cx, cy + 10, text=self.camera_status_message,
                fill=COLOR_TEXT_WHITE, font=("Segoe UI", 11)
            )
            self.video_canvas.create_text(
                cx, cy + 40, text="Attempting automatic hardware reconnection. Use fallback controls below if needed.",
                fill=COLOR_TEXT_DIM, font=("Segoe UI", 9)
            )
        elif self.camera_state == CameraState.NO_CAMERA:
            self.video_canvas.create_text(
                cx, cy - 45, text="⚠️ No Camera Could Be Connected",
                fill=COLOR_DANGER, font=("Segoe UI", 16, "bold")
            )
            self.video_canvas.create_text(
                cx, cy - 10, text="No accessible webcam detected or camera permission denied by OS.",
                fill=COLOR_TEXT_WHITE, font=("Segoe UI", 11)
            )
            self.video_canvas.create_text(
                cx, cy + 25, text="Select an option below to proceed:",
                fill=COLOR_TEXT_MUTED, font=("Segoe UI", 10)
            )
            self.video_canvas.create_text(
                cx, cy + 55, text="[Retry Camera]   •   [Select Camera]   •   [Use Video File]",
                fill=COLOR_ACCENT, font=("Segoe UI", 11, "bold")
            )
        elif self.camera_state == CameraState.DISCONNECTED:
            self.video_canvas.create_text(
                cx, cy - 25, text="📹 Camera Feed Disconnected",
                fill=COLOR_TEXT_MUTED, font=("Segoe UI", 15, "bold")
            )
            self.video_canvas.create_text(
                cx, cy + 15, text="Click 'Start Camera' or 'Load Video' to begin monitoring.",
                fill=COLOR_TEXT_DIM, font=("Segoe UI", 10)
            )
        elif self.camera_state == CameraState.ERROR:
            self.video_canvas.create_text(
                cx, cy - 25, text="❌ Camera Error",
                fill=COLOR_DANGER, font=("Segoe UI", 15, "bold")
            )
            self.video_canvas.create_text(
                cx, cy + 15, text=self.camera_status_message[:80],
                fill=COLOR_TEXT_MUTED, font=("Segoe UI", 10)
            )

    # -----------------------------------------------------------------------
    # External Pipeline Integration Callbacks (Safe GUI Thread Dispatch)
    # -----------------------------------------------------------------------
    def update_frame(self, frame) -> None:
        """Receive annotated frame from VideoProcessor (thread-safe)."""
        self.current_frame = frame
        if not self.running:
            self.running = True
            self._update_button_states()

    def update_pipeline_result(self, result: Any) -> None:
        """
        Receive complete ProcessingResult from VideoProcessor and update:
        - People active tracks table
        - Anomaly log
        - Metric cards
        """
        # Always update StringVars immediately (thread-safe in Python Tkinter)
        if hasattr(result, "inference_time_ms") and result.inference_time_ms > 0:
            self.status_vars["inference_time"].set(f"{result.inference_time_ms:.1f}ms")
        if hasattr(result, "process_time_ms") and result.process_time_ms > 0:
            self.status_vars["process_time"].set(f"{result.process_time_ms:.1f}ms")
        if hasattr(result, "tracks"):
            self.status_vars["active_tracks"].set(str(len(result.tracks)))
        if hasattr(result, "detections"):
            self.status_vars["objects_count"].set(str(len(result.detections)))

        def _apply():
            try:
                # Update People Treeview
                if hasattr(result, "tracks") and result.tracks:
                    existing = self.people_tree.get_children()
                    for item in existing:
                        self.people_tree.delete(item)

                    for track in result.tracks:
                        act = result.activities.get(track.track_id).value if hasattr(result, "activities") and track.track_id in result.activities else "Unknown"
                        posture = getattr(track, "posture", "N/A")
                        gesture = getattr(track, "gesture", "N/A")
                        face = "Yes" if getattr(track, "face_detected", False) else "No"
                        bbox_str = f"({int(track.bbox[0])}, {int(track.bbox[1])}, {int(track.bbox[2])}, {int(track.bbox[3])})"
                        values = (
                            f"ID:{track.track_id}",
                            act,
                            posture,
                            gesture,
                            face,
                            f"{track.speed:.1f}",
                            f"{track.total_distance:.1f}",
                            f"{track.duration:.1f}",
                            bbox_str,
                            "Tracking"
                        )
                        self.people_tree.insert("", tk.END, values=values)

                    self.people_summary_lbl.configure(
                        text=f"Active Tracks: {len(result.tracks)} | Total Detected: {self.stats_vars['total_detected'].get()}"
                    )

                # Update Anomalies View
                if hasattr(result, "anomalies") and result.anomalies:
                    for a in result.anomalies:
                        if getattr(a, "anomaly_type", None) and a.anomaly_type.value == "ANOMALY":
                            ts = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
                            vals = (ts, f"ID:{a.track_id}", f"{a.score:.3f}", "ANOMALY", "Irregular behavior detected")
                            self.anomaly_tree.insert("", 0, values=vals)
                            # Cap table
                            children = self.anomaly_tree.get_children()
                            if len(children) > 100:
                                self.anomaly_tree.delete(children[-1])

                if hasattr(result, "tracks") and "active_tracks" in self.metric_cards:
                    self.metric_cards["active_tracks"].set_value(str(len(result.tracks)))

                if hasattr(result, "detections") and "objects" in self.metric_cards:
                    self.metric_cards["objects"].set_value(str(len(result.detections)))

                # Check and update posture mode availability
                self._update_posture_mode_availability(result)

            except Exception as e:
                logger.debug(f"Pipeline result GUI dispatch suppressed: {e}")

        if getattr(self, "_pipeline_update_pending", False):
            return
        self._pipeline_update_pending = True

        def _apply_wrapper():
            try:
                _apply()
            finally:
                self._pipeline_update_pending = False

        try:
            if threading.current_thread() is threading.main_thread():
                _apply_wrapper()
            else:
                self.root.after(0, _apply_wrapper)
        except Exception:
            self._pipeline_update_pending = False

    def _update_posture_mode_availability(self, result: Any) -> None:
        """Update Posture, Hand Gesture and Face mode button availability based on estimator status."""
        pose_available = False
        gesture_available = False
        face_available = False
        if hasattr(result, "poses") and result.poses is not None:
            pose_available = True
            # Check if gesture analyzer is enabled
            if hasattr(result, "pose_model_available"):
                gesture_available = result.pose_model_available
            else:
                gesture_available = True
        elif hasattr(result, "pose_model_available"):
            pose_available = result.pose_model_available
            gesture_available = result.pose_model_available

        # Check face detection availability
        if hasattr(result, "face_model_available"):
            face_available = result.face_model_available
        elif hasattr(result, "faces") and result.faces is not None:
            face_available = True

        # Update Posture mode button
        pose_changed = False
        if not hasattr(self, "_pose_available") or self._pose_available != pose_available:
            self._pose_available = pose_available
            pose_changed = True

        if pose_changed and "Posture" in self.mode_buttons:
            btn = self.mode_buttons["Posture"]
            if pose_available:
                btn.configure(
                    fg=COLOR_TEXT_WHITE,
                    font=("Segoe UI", 8, "bold"),
                    text="Posture",
                    state=tk.NORMAL
                )
            else:
                btn.configure(
                    fg=COLOR_TEXT_DIM,
                    font=("Segoe UI", 8),
                    text="Posture (Upcoming: UPGRADE 7)",
                    state=tk.DISABLED
                )

        # Update Hand Gesture mode button
        gesture_changed = False
        if not hasattr(self, "_gesture_available") or self._gesture_available != gesture_available:
            self._gesture_available = gesture_available
            gesture_changed = True

        if gesture_changed and "Hand Gesture" in self.mode_buttons:
            btn = self.mode_buttons["Hand Gesture"]
            if gesture_available:
                btn.configure(
                    fg=COLOR_TEXT_WHITE,
                    font=("Segoe UI", 8, "bold"),
                    text="Hand Gesture",
                    state=tk.NORMAL
                )
            else:
                btn.configure(
                    fg=COLOR_TEXT_DIM,
                    font=("Segoe UI", 8),
                    text="Hand Gesture (Upcoming: UPGRADE 8)",
                    state=tk.DISABLED
                )

        # Update Face mode button
        face_changed = False
        if not hasattr(self, "_face_available") or self._face_available != face_available:
            self._face_available = face_available
            face_changed = True

        if face_changed and "Face" in self.mode_buttons:
            btn = self.mode_buttons["Face"]
            if face_available:
                btn.configure(
                    fg=COLOR_TEXT_WHITE,
                    font=("Segoe UI", 8, "bold"),
                    text="Face",
                    state=tk.NORMAL
                )
            else:
                btn.configure(
                    fg=COLOR_TEXT_DIM,
                    font=("Segoe UI", 8),
                    text="Face (Upcoming: UPGRADE 9)",
                    state=tk.DISABLED
                )

    def set_camera_state(self, state: CameraState, message: str = "") -> None:
        """Thread-safe camera state transition handler."""
        self.camera_state = state
        self.camera_status_message = message or state.value

        def _apply():
            state_display = {
                CameraState.SEARCHING: ("SEARCHING", COLOR_WARNING),
                CameraState.CONNECTING: ("CONNECTING", COLOR_WARNING),
                CameraState.CONNECTED: ("CONNECTED", COLOR_SUCCESS),
                CameraState.NO_CAMERA: ("NO CAMERA", COLOR_DANGER),
                CameraState.DISCONNECTED: ("DISCONNECTED", COLOR_TEXT_MUTED),
                CameraState.RECONNECTING: ("RECONNECTING", COLOR_ORANGE),
                CameraState.ERROR: ("ERROR", COLOR_DANGER),
            }

            text, color = state_display.get(state, (state.value, COLOR_TEXT_WHITE))
            self.status_vars["camera"].set(text)

            # Update Metric Card
            if "camera_status" in self.metric_cards:
                self.metric_cards["camera_status"].set_value(text, color=color, subtitle=message[:22] if message else "")

            # Update Top Bar Pill
            if state == CameraState.CONNECTED:
                self.top_status_pill.configure(text=f"● {text}", fg=COLOR_SUCCESS)
                self.running = True
            elif state in (CameraState.SEARCHING, CameraState.CONNECTING):
                self.top_status_pill.configure(text=f"◌ {text}", fg=COLOR_WARNING)
            elif state == CameraState.RECONNECTING:
                self.top_status_pill.configure(text=f"🔄 {text}", fg=COLOR_ORANGE)
                self.running = False
            elif state in (CameraState.NO_CAMERA, CameraState.ERROR):
                self.top_status_pill.configure(text=f"⚠️ {text}", fg=COLOR_DANGER)
                self.running = False
            else:
                self.top_status_pill.configure(text=f"○ {text}", fg=COLOR_TEXT_MUTED)
                self.running = False

            if message:
                self.set_status_bar(message)
                self.video_info_var.set(message)

            if hasattr(self, "live_camera_title_lbl"):
                src = self.source_description if self.source_description != "No Source" else "No Source"
                self.live_camera_title_lbl.configure(text=f"LIVE CAMERA: {src}")

            self._update_button_states()

        try:
            if threading.current_thread() is threading.main_thread():
                _apply()
            else:
                self.root.after(0, _apply)
        except Exception:
            try:
                _apply()
            except Exception:
                pass

    def update_camera_list(self, cameras: List[CameraDeviceInfo]) -> None:
        """Update camera selector dropdown safely."""
        def _apply():
            self.camera_devices = cameras
            if cameras:
                names = [f"{c.device_id}: {c.display_name()}" for c in cameras]
                self.camera_combobox["values"] = names
                default_cam = next((c for c in cameras if c.is_default), cameras[0])
                default_name = f"{default_cam.device_id}: {default_cam.display_name()}"
                self.camera_selector_var.set(default_name)
                self.selected_camera_id = default_cam.device_id
                self.source_description = default_cam.name
                self.top_source_badge.configure(text=f"SRC: {default_cam.name[:16]}")
                if hasattr(self, "live_camera_title_lbl"):
                    self.live_camera_title_lbl.configure(text=f"LIVE CAMERA: {default_cam.name}")
            else:
                self.camera_combobox["values"] = ["No camera detected"]
                self.camera_selector_var.set("No camera detected")
                self.top_source_badge.configure(text="SRC: NONE")
                if hasattr(self, "live_camera_title_lbl"):
                    self.live_camera_title_lbl.configure(text="LIVE CAMERA: No Source")

        try:
            if threading.current_thread() is threading.main_thread():
                _apply()
            else:
                self.root.after(0, _apply)
        except Exception:
            try:
                _apply()
            except Exception:
                pass

    def _on_combobox_camera_selected(self, event=None) -> None:
        """User switched camera candidate."""
        sel = self.camera_selector_var.get()
        if ":" in sel:
            try:
                dev_id = int(sel.split(":")[0].strip())
                self.selected_camera_id = dev_id
                if self.on_select_camera:
                    self.set_status_bar(f"Switching to camera {dev_id}...")
                    self.on_select_camera(dev_id)
            except ValueError:
                pass

    def update_status(self, **kwargs) -> None:
        """Update telemetry values and metric cards (thread-safe)."""
        def _apply():
            for key, value in kwargs.items():
                if key in self.status_vars:
                    self.status_vars[key].set(str(value))

            # Update metric cards
            if "people_count" in kwargs and "people" in self.metric_cards:
                self.metric_cards["people"].set_value(str(kwargs["people_count"]))

            if "fps" in kwargs:
                fps_str = str(kwargs["fps"])
                self.top_fps_badge.configure(text=f"FPS: {fps_str}")
                if "fps" in self.metric_cards:
                    self.metric_cards["fps"].set_value(fps_str)

            if "current_activity" in kwargs and "current_activity" in self.metric_cards:
                self.metric_cards["current_activity"].set_value(str(kwargs["current_activity"]))

            if "anomaly_status" in kwargs and "anomalies" in self.metric_cards:
                status = kwargs["anomaly_status"]
                color = COLOR_DANGER if status == "ANOMALY" else COLOR_SUCCESS
                self.metric_cards["anomalies"].set_value(status, color=color)

            # Update Live Monitor subsystem status badges
            if "pipeline_detection" in kwargs and hasattr(self, "badge_det"):
                det_val = kwargs["pipeline_detection"]
                color = COLOR_SUCCESS if det_val == "ACTIVE" else (COLOR_DANGER if det_val == "ERROR" else COLOR_TEXT_DIM)
                self.badge_det.configure(text=f"DET: {det_val}", fg=color)

            if "pipeline_tracking" in kwargs and hasattr(self, "badge_trk"):
                trk_val = kwargs["pipeline_tracking"]
                color = COLOR_SUCCESS if trk_val == "ACTIVE" else (COLOR_DANGER if trk_val == "ERROR" else COLOR_TEXT_DIM)
                self.badge_trk.configure(text=f"TRK: {trk_val}", fg=color)

            if "pipeline_activity" in kwargs and hasattr(self, "badge_act"):
                act_val = kwargs["pipeline_activity"]
                color = COLOR_SUCCESS if act_val == "ACTIVE" else COLOR_TEXT_DIM
                self.badge_act.configure(text=f"ACT: {act_val}", fg=color)

            if "pipeline_anomaly" in kwargs and hasattr(self, "badge_anom"):
                anom_val = kwargs["pipeline_anomaly"]
                color = COLOR_SUCCESS if anom_val == "ACTIVE" else COLOR_TEXT_DIM
                self.badge_anom.configure(text=f"ANOM: {anom_val}", fg=color)

            # Update Live Monitor top telemetry label
            if hasattr(self, "live_telemetry_top_lbl"):
                res_val = self.status_vars.get("resolution", tk.StringVar(value="N/A")).get()
                fps_val = self.status_vars.get("fps", tk.StringVar(value="0.0")).get()
                inf_val = self.status_vars.get("inference_time", tk.StringVar(value="N/A")).get()
                self.live_telemetry_top_lbl.configure(text=f"RES: {res_val} | FPS: {fps_val} | INFER: {inf_val}")

            # Update bottom video telemetry strip
            if hasattr(self, "video_telemetry_var"):
                fps_txt = self.status_vars.get("fps", tk.StringVar(value="0.0")).get()
                inf_txt = self.status_vars.get("inference_time", tk.StringVar(value="N/A")).get()
                people_txt = self.status_vars.get("people_count", tk.StringVar(value="0")).get()
                tracks_txt = self.status_vars.get("active_tracks", tk.StringVar(value="0")).get()
                cam_txt = self.status_vars.get("camera", tk.StringVar(value="DISCONNECTED")).get()
                self.video_telemetry_var.set(
                    f"FPS: {fps_txt} | Infer: {inf_txt} | People: {people_txt} | Tracks: {tracks_txt} | State: {cam_txt}"
                )

            # Update Live Monitor camera title
            if hasattr(self, "live_camera_title_lbl"):
                src = self.source_description if self.source_description != "No Source" else "No Source"
                self.live_camera_title_lbl.configure(text=f"LIVE CAMERA: {src}")

        try:
            if threading.current_thread() is threading.main_thread():
                _apply()
            else:
                self.root.after(0, _apply)
        except Exception:
            pass

    def update_stats(self, **kwargs) -> None:
        """Update running statistics values (thread-safe)."""
        def _apply():
            for key, value in kwargs.items():
                if key in self.stats_vars:
                    self.stats_vars[key].set(str(value))

            if "total_detected" in kwargs and "total_detected" in getattr(self, "stat_cards", {}):
                self.stat_cards["total_detected"].set_value(str(kwargs["total_detected"]))

            if "total_entries" in kwargs and "total_entries" in getattr(self, "stat_cards", {}):
                self.stat_cards["total_entries"].set_value(str(kwargs["total_entries"]))

            if "total_exits" in kwargs and "total_exits" in getattr(self, "stat_cards", {}):
                self.stat_cards["total_exits"].set_value(str(kwargs["total_exits"]))

            if "anomaly_count" in kwargs and "anomaly_count" in getattr(self, "stat_cards", {}):
                self.stat_cards["anomaly_count"].set_value(str(kwargs["anomaly_count"]))

            if "walking_count" in kwargs and "walking" in getattr(self, "act_cards", {}):
                self.act_cards["walking"].set_value(str(kwargs["walking_count"]))

            if "standing_count" in kwargs and "standing" in getattr(self, "act_cards", {}):
                self.act_cards["standing"].set_value(str(kwargs["standing_count"]))

            if "running_count" in kwargs and "running" in getattr(self, "act_cards", {}):
                self.act_cards["running"].set_value(str(kwargs["running_count"]))

            if "loitering_events" in kwargs and "loitering" in getattr(self, "act_cards", {}):
                self.act_cards["loitering"].set_value(str(kwargs["loitering_events"]))

            if "restricted_zone_events" in kwargs and "zones" in getattr(self, "act_cards", {}):
                self.act_cards["zones"].set_value(str(kwargs["restricted_zone_events"]))

        try:
            if threading.current_thread() is threading.main_thread():
                _apply()
            else:
                self.root.after(0, _apply)
        except Exception:
            pass

    def add_event(self, event_data: dict) -> None:
        """Record event into real-time surveillance event table (thread-safe)."""
        def _apply():
            try:
                values = (
                    event_data.get("timestamp", time.strftime("%H:%M:%S")),
                    event_data.get("event_type", "EVENT"),
                    event_data.get("person_id", "N/A"),
                    event_data.get("activity", "N/A"),
                    event_data.get("zone", "General"),
                    event_data.get("details", "")
                )
                self.events_tree.insert("", 0, values=values)

                # Update preview label on Dashboard view
                preview_txt = f"[{values[0]}] {values[1]} | {values[2]} | {values[3]} | {values[5]}"
                if hasattr(self, "dash_recent_event_lbl"):
                    self.dash_recent_event_lbl.configure(text=preview_txt)

                children = self.events_tree.get_children()
                if len(children) > 150:
                    self.events_tree.delete(children[-1])

                # Also update Live Monitor live_events_tree
                if hasattr(self, "live_events_tree"):
                    live_vals = (values[0], values[2], values[1], values[5][:35])
                    self.live_events_tree.insert("", 0, values=live_vals)
                    live_children = self.live_events_tree.get_children()
                    if len(live_children) > 30:
                        self.live_events_tree.delete(live_children[-1])
            except Exception as e:
                logger.error(f"Error adding event: {e}")

        try:
            if threading.current_thread() is threading.main_thread():
                _apply()
            else:
                self.root.after(0, _apply)
        except Exception:
            pass

    def clear_events(self) -> None:
        """Clear all events from the surveillance log table and live monitor stream."""
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
        if hasattr(self, "live_events_tree"):
            for item in self.live_events_tree.get_children():
                self.live_events_tree.delete(item)
        if hasattr(self, "dash_recent_event_lbl"):
            self.dash_recent_event_lbl.configure(text="No events recorded. Event log cleared.")

    def set_status_bar(self, message: str) -> None:
        """Set message in bottom status bar."""
        self.status_bar_var.set(message)

    # -----------------------------------------------------------------------
    # Button State Synchronizer
    # -----------------------------------------------------------------------
    def _update_button_states(self) -> None:
        """Synchronize button enabled/disabled states across all views."""
        is_running = self.running

        # Sidebar buttons
        self.side_start_btn.configure(state=tk.NORMAL if not is_running else tk.DISABLED)
        self.side_stop_btn.configure(state=tk.NORMAL if is_running else tk.DISABLED)

        # Video control buttons
        if "start_btn" in self.control_buttons:
            self.control_buttons["start_btn"].configure(state=tk.NORMAL if not is_running else tk.DISABLED)
        if "stop_btn" in self.control_buttons:
            self.control_buttons["stop_btn"].configure(state=tk.NORMAL if is_running else tk.DISABLED)
        if "pause_btn" in self.control_buttons:
            self.control_buttons["pause_btn"].configure(state=tk.NORMAL if is_running and not self.paused else tk.DISABLED)
        if "resume_btn" in self.control_buttons:
            self.control_buttons["resume_btn"].configure(state=tk.NORMAL if is_running and self.paused else tk.DISABLED)

    # -----------------------------------------------------------------------
    # Asynchronous Camera Discovery
    # -----------------------------------------------------------------------
    def start_auto_discovery(self) -> None:
        """Trigger camera discovery on background thread without freezing GUI."""
        def _worker():
            self.set_camera_state(CameraState.SEARCHING, "Probing available video devices...")
            if self.on_auto_start_camera:
                self.on_auto_start_camera()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # -----------------------------------------------------------------------
    # Button Click Actions
    # -----------------------------------------------------------------------
    def _on_retry_click(self) -> None:
        self.set_status_bar("Retrying camera discovery...")
        if self.on_retry_camera:
            self.on_retry_camera()
        else:
            self.start_auto_discovery()

    def _on_select_camera_dialog(self) -> None:
        """Open camera selection dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Camera Device")
        dialog.geometry("460x220")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Select Camera Candidate:", font=("Segoe UI", 10, "bold")).pack(pady=10)

        options = [f"{c.device_id}: {c.display_name()}" for c in self.camera_devices]
        if not options:
            options = ["0: Default Device Index 0", "1: Device Index 1"]

        selected_var = tk.StringVar(value=options[0])
        combo = ttk.Combobox(dialog, textvariable=selected_var, values=options, state="readonly", width=42)
        combo.pack(pady=10)

        def connect():
            val = selected_var.get()
            dev_id = 0
            if ":" in val:
                try:
                    dev_id = int(val.split(":")[0].strip())
                except ValueError:
                    dev_id = 0
            dialog.destroy()
            if self.on_select_camera:
                self.on_select_camera(dev_id)
            elif self.on_start_camera:
                self.on_start_camera(dev_id)

        ttk.Button(dialog, text="Connect Selected Camera", command=connect).pack(pady=15)

    def _on_start_click(self) -> None:
        if self.on_start_camera:
            dev_id = self.selected_camera_id if self.selected_camera_id is not None else 0
            self.set_status_bar(f"Starting camera {dev_id}...")
            result = self.on_start_camera(dev_id)
            if result is False:
                self.running = False
                self._update_button_states()
                self.set_status_bar("Camera could not be started")

    def _on_stop_click(self) -> None:
        self.running = False
        self.paused = False
        self._update_button_states()
        self.set_status_bar("Camera stopped")
        if self.on_stop_camera:
            self.on_stop_camera()
        self.current_frame = None
        self.set_camera_state(CameraState.DISCONNECTED, "Camera stopped")
        self.status_vars["fps"].set("0.0")
        self.status_vars["people_count"].set("0")
        self.top_fps_badge.configure(text="FPS: 0.0")
        self.video_canvas.delete("all")
        self._render_canvas_placeholder()

    def _on_pause_click(self) -> None:
        if self.on_pause:
            self.paused = True
            self._update_button_states()
            self.set_status_bar("Paused")
            self.on_pause()

    def _on_resume_click(self) -> None:
        if self.on_resume:
            self.paused = False
            self._update_button_states()
            self.set_status_bar("Resumed")
            self.on_resume()

    def _on_load_video_click(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select CCTV Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.set_status_bar(f"Loading video: {Path(file_path).name}")
            self.source_description = Path(file_path).name
            self.top_source_badge.configure(text=f"SRC: {self.source_description[:16]}")
            if self.on_load_video:
                self.running = True
                self.on_load_video(file_path)
                self.show_view("live_monitor")

    def _on_connect_ip_click(self) -> None:
        self.show_view("camera")

    def _on_screenshot_click(self) -> None:
        if self.on_screenshot:
            path = self.on_screenshot()
            msg = f"Screenshot saved: {path}" if path else "Screenshot saved"
            self.set_status_bar(msg)

    def _on_export_report_click(self) -> None:
        if self.on_export_report:
            self.set_status_bar("Exporting surveillance report...")
            path = self.on_export_report()
            msg = f"Report exported: {path}" if path else "Report generated"
            self.set_status_bar(msg)
            messagebox.showinfo("Export Complete", f"Surveillance report successfully exported:\n{path or 'Done'}")

    def _on_clear_events_click(self) -> None:
        if messagebox.askyesno("Clear Events", "Clear all recorded events from the surveillance log?"):
            self.clear_events()
            if self.on_clear_events:
                self.on_clear_events()
            self.set_status_bar("Events cleared")

    def _on_settings_click(self) -> None:
        self.show_view("settings")
        if self.on_settings:
            self.on_settings()
        self.set_status_bar("Settings view opened")

    # -----------------------------------------------------------------------
    # Periodic GUI Update Loops (Clock, FPS, Video Display)
    # -----------------------------------------------------------------------
    def _start_gui_loops(self) -> None:
        """Start periodic GUI update loops."""
        self._update_clock()
        self._update_fps()
        self._update_gui()

    def _update_clock(self) -> None:
        """Update top bar digital clock every second."""
        curr_time = time.strftime("%H:%M:%S")
        self.top_clock_label.configure(text=curr_time)
        self.root.after(1000, self._update_clock)

    def _update_fps(self) -> None:
        """Calculate and update FPS counter."""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
            if self.running:
                fps_txt = f"{self.current_fps:.1f}"
                self.status_vars["fps"].set(fps_txt)
                self.top_fps_badge.configure(text=f"FPS: {fps_txt}")
                if "fps" in self.metric_cards:
                    self.metric_cards["fps"].set_value(fps_txt)

        self.root.after(100, self._update_fps)

    def _update_gui(self) -> None:
        """Periodic video canvas refresh."""
        if self.running and not self.paused:
            self._update_video_display()
        elif not self.running:
            self._render_canvas_placeholder()

        self.root.after(self.config.gui.update_interval_ms, self._update_gui)

    # -----------------------------------------------------------------------
    # Application Run and Clean Shutdown
    # -----------------------------------------------------------------------
    def run(self) -> None:
        """Start the Tkinter main event loop."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _on_closing(self) -> None:
        """Handle window close event with clean pipeline shutdown."""
        if self.running:
            if messagebox.askokcancel("Quit", "Live Activity Analyzer is currently monitoring. Stop and exit?"):
                if self.on_stop_camera:
                    self.on_stop_camera()
                self.root.destroy()
        else:
            self.root.destroy()


def launch_gui(config_manager: ConfigManager) -> ApplicationGUI:
    """Launch the GUI application."""
    app = ApplicationGUI(config_manager)
    return app