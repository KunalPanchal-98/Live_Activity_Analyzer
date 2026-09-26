"""
Main GUI Application for Live Activity Analyzer.

Tkinter-based desktop dashboard with video display, controls,
statistics panels, and event monitoring.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable
from PIL import Image, ImageTk

from app.config_manager import ConfigManager, AppConfig


logger = logging.getLogger(__name__)


class ApplicationGUI:
    """
    Main Tkinter application window.
    
    Provides a responsive dashboard with:
    - Live video feed display
    - System status and metrics
    - Control buttons (Start/Stop/Pause/Resume)
    - Event log and alerts
    - Statistics and charts
    - Settings dialog
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config: AppConfig = config_manager.get_config()
        
        self.root = tk.Tk()
        self.root.title("Live Activity Analyzer - AI-Based Real-Time CCTV Activity Analysis")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # State
        self.running = False
        self.paused = False
        self.current_frame = None
        self.photo_image = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Callbacks (to be connected by main application)
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
        
        self._setup_styles()
        self._create_widgets()
        self._setup_layout()
        self._start_gui_update_loop()
        
        logger.info("GUI initialized")
    
    def _setup_styles(self) -> None:
        """Configure ttk styles for a professional appearance."""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure colors for dark theme
        bg_color = "#2b2b2b"
        fg_color = "#ffffff"
        accent_color = "#0078d4"
        panel_bg = "#333333"
        
        self.root.configure(bg=bg_color)
        
        style.configure(".", background=bg_color, foreground=fg_color)
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TButton", background=panel_bg, foreground=fg_color)
        style.configure("TLabelFrame", background=bg_color, foreground=fg_color)
        style.configure("TLabelFrame.Label", background=bg_color, foreground=fg_color)
        style.configure("TEntry", fieldbackground=panel_bg, foreground=fg_color)
        style.configure("TCombobox", fieldbackground=panel_bg, foreground=fg_color)
        style.configure("Treeview", background=panel_bg, foreground=fg_color, fieldbackground=panel_bg)
        style.configure("Treeview.Heading", background=accent_color, foreground=fg_color)
        style.configure("Horizontal.TProgressbar", background=accent_color)
        
        style.map("TButton",
            background=[("active", accent_color), ("pressed", "#005a9e")],
            foreground=[("active", fg_color), ("pressed", fg_color)]
        )
    
    def _create_widgets(self) -> None:
        """Create all GUI widgets."""
        # Main containers
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.left_frame = ttk.Frame(self.main_paned)
        self.right_frame = ttk.Frame(self.main_paned)
        
        # Left side: Video display
        self._create_video_panel()
        
        # Right side: Status, stats, controls
        self._create_status_panel()
        self._create_controls_panel()
        self._create_events_panel()
        self._create_stats_panel()
        
        # Bottom: Status bar
        self._create_status_bar()
    
    def _create_video_panel(self) -> None:
        """Create video display panel."""
        video_frame = ttk.LabelFrame(self.left_frame, text="Live Video Feed", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video canvas
        self.video_canvas = tk.Canvas(
            video_frame,
            bg="#1a1a1a",
            highlightthickness=0
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Video info overlay
        self.video_info_var = tk.StringVar(value="No camera connected")
        self.video_info_label = ttk.Label(video_frame, textvariable=self.video_info_var, font=("Consolas", 9))
        self.video_info_label.pack(anchor=tk.W, pady=(2, 0))
    
    def _create_status_panel(self) -> None:
        """Create system status panel."""
        status_frame = ttk.LabelFrame(self.right_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Status grid
        self.status_vars = {
            "camera": tk.StringVar(value="Disconnected"),
            "fps": tk.StringVar(value="0.0"),
            "people_count": tk.StringVar(value="0"),
            "current_activity": tk.StringVar(value="None"),
            "anomaly_status": tk.StringVar(value="NORMAL"),
            "alert_status": tk.StringVar(value="No Alerts"),
        }
        
        labels = [
            ("Camera:", "camera"),
            ("FPS:", "fps"),
            ("People Detected:", "people_count"),
            ("Current Activity:", "current_activity"),
            ("Anomaly Status:", "anomaly_status"),
            ("Alert Status:", "alert_status"),
        ]
        
        for i, (label_text, key) in enumerate(labels):
            ttk.Label(status_frame, text=label_text, font=("Segoe UI", 10, "bold")).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=3
            )
            value_label = ttk.Label(status_frame, textvariable=self.status_vars[key], font=("Segoe UI", 10))
            value_label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)
            
            # Color coding for anomaly status
            if key == "anomaly_status":
                self.anomaly_label = value_label
            if key == "alert_status":
                self.alert_label = value_label
        
        status_frame.columnconfigure(1, weight=1)
    
    def _create_controls_panel(self) -> None:
        """Create control buttons panel."""
        controls_frame = ttk.LabelFrame(self.right_frame, text="Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        buttons = [
            ("Start Camera", self._on_start_click, "start_btn"),
            ("Stop Camera", self._on_stop_click, "stop_btn"),
            ("Pause", self._on_pause_click, "pause_btn"),
            ("Resume", self._on_resume_click, "resume_btn"),
            ("Load Video", self._on_load_video_click, "load_btn"),
            ("Connect IP Camera", self._on_connect_ip_click, "ip_btn"),
            ("Screenshot", self._on_screenshot_click, "screenshot_btn"),
            ("Export Report", self._on_export_report_click, "export_btn"),
            ("Clear Events", self._on_clear_events_click, "clear_btn"),
            ("Settings", self._on_settings_click, "settings_btn"),
        ]
        
        self.control_buttons = {}
        for i, (text, command, key) in enumerate(buttons):
            btn = ttk.Button(controls_frame, text=text, command=command, width=20)
            btn.grid(row=i // 2, column=i % 2, padx=5, pady=3, sticky=tk.EW)
            self.control_buttons[key] = btn
        
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        
        # Initial button states
        self._update_button_states()
    
    def _create_events_panel(self) -> None:
        """Create event log panel."""
        events_frame = ttk.LabelFrame(self.right_frame, text="Event Log", padding=5)
        events_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for events
        columns = ("Time", "Type", "Person ID", "Activity", "Zone", "Details")
        self.events_tree = ttk.Treeview(events_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.events_tree.heading(col, text=col)
            self.events_tree.column(col, width=100, anchor=tk.W)
        
        self.events_tree.column("Time", width=130)
        self.events_tree.column("Details", width=200)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(events_frame, orient=tk.VERTICAL, command=self.events_tree.yview)
        self.events_tree.configure(yscrollcommand=scrollbar.set)
        
        self.events_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_stats_panel(self) -> None:
        """Create statistics panel."""
        stats_frame = ttk.LabelFrame(self.right_frame, text="Statistics", padding=5)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_vars = {
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
        
        stats_layout = [
            ("Total Detected:", "total_detected"),
            ("Entries:", "total_entries"),
            ("Exits:", "total_exits"),
            ("Walking:", "walking_count"),
            ("Standing:", "standing_count"),
            ("Running:", "running_count"),
            ("Loitering Events:", "loitering_events"),
            ("Anomalies:", "anomaly_count"),
            ("Zone Events:", "restricted_zone_events"),
        ]
        
        for i, (label_text, key) in enumerate(stats_layout):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(stats_frame, text=label_text, font=("Segoe UI", 9)).grid(
                row=row, column=col, sticky=tk.W, padx=5, pady=2
            )
            ttk.Label(stats_frame, textvariable=self.stats_vars[key], font=("Segoe UI", 9, "bold")).grid(
                row=row, column=col + 1, sticky=tk.W, padx=5, pady=2
            )
    
    def _create_status_bar(self) -> None:
        """Create bottom status bar."""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        self.status_bar_var = tk.StringVar(value="Ready")
        ttk.Label(self.status_bar, textvariable=self.status_bar_var).pack(side=tk.LEFT)
        
        # Version info
        ttk.Label(self.status_bar, text="Live Activity Analyzer v1.0.0", font=("Segoe UI", 8)).pack(side=tk.RIGHT)
    
    def _setup_layout(self) -> None:
        """Configure main window layout."""
        self.main_paned.add(self.left_frame, weight=3)
        self.main_paned.add(self.right_frame, weight=2)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _start_gui_update_loop(self) -> None:
        """Start periodic GUI update loop."""
        self._update_fps()
        self._update_gui()
    
    def _update_gui(self) -> None:
        """Periodic GUI update."""
        if self.running and not self.paused:
            self._update_video_display()
        
        self.root.after(self.config.gui.update_interval_ms, self._update_gui)
    
    def _update_fps(self) -> None:
        """Calculate and update FPS."""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
            self.status_vars["fps"].set(f"{self.current_fps:.1f}")
        
        self.root.after(100, self._update_fps)
    
    def _update_video_display(self) -> None:
        """Update video canvas with current frame."""
        if self.current_frame is not None:
            try:
                # Resize frame to fit canvas
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    frame_rgb = self.current_frame
                    h, w = frame_rgb.shape[:2]
                    
                    # Calculate scaling
                    scale = min(canvas_width / w, canvas_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    # Resize
                    import cv2
                    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Convert to PIL and then PhotoImage
                    image = Image.fromarray(resized)
                    self.photo_image = ImageTk.PhotoImage(image)
                    
                    # Update canvas
                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(
                        canvas_width // 2, canvas_height // 2,
                        image=self.photo_image, anchor=tk.CENTER
                    )
            except Exception as e:
                logger.debug(f"Video display error: {e}")
    
    def update_frame(self, frame) -> None:
        """Update current frame from video processor (called from processing thread)."""
        self.current_frame = frame
    
    def update_status(self, **kwargs) -> None:
        """Update status panel values."""
        for key, value in kwargs.items():
            if key in self.status_vars:
                self.status_vars[key].set(str(value))
        
        # Color coding for anomaly status
        if "anomaly_status" in kwargs:
            status = kwargs["anomaly_status"]
            if hasattr(self, "anomaly_label"):
                if status == "ANOMALY":
                    self.anomaly_label.configure(foreground="#ff4444")
                else:
                    self.anomaly_label.configure(foreground="#44ff44")
        
        if "alert_status" in kwargs:
            status = kwargs["alert_status"]
            if hasattr(self, "alert_label"):
                if "Alert" in status and "No" not in status:
                    self.alert_label.configure(foreground="#ffaa00")
                else:
                    self.alert_label.configure(foreground="#44ff44")
    
    def update_stats(self, **kwargs) -> None:
        """Update statistics panel values."""
        for key, value in kwargs.items():
            if key in self.stats_vars:
                self.stats_vars[key].set(str(value))
    
    def add_event(self, event_data: dict) -> None:
        """Add event to the event log."""
        try:
            values = (
                event_data.get("timestamp", ""),
                event_data.get("event_type", ""),
                event_data.get("person_id", ""),
                event_data.get("activity", ""),
                event_data.get("zone", ""),
                event_data.get("details", "")
            )
            self.events_tree.insert("", 0, values=values)
            
            # Limit events in tree
            children = self.events_tree.get_children()
            if len(children) > 100:
                self.events_tree.delete(children[-1])
        except Exception as e:
            logger.error(f"Error adding event: {e}")
    
    def clear_events(self) -> None:
        """Clear all events from the log."""
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
    
    def set_status_bar(self, message: str) -> None:
        """Set status bar message."""
        self.status_bar_var.set(message)
    
    def _update_button_states(self) -> None:
        """Update control button enabled/disabled states."""
        self.control_buttons["start_btn"].configure(state=tk.NORMAL if not self.running else tk.DISABLED)
        self.control_buttons["stop_btn"].configure(state=tk.NORMAL if self.running else tk.DISABLED)
        self.control_buttons["pause_btn"].configure(state=tk.NORMAL if self.running and not self.paused else tk.DISABLED)
        self.control_buttons["resume_btn"].configure(state=tk.NORMAL if self.running and self.paused else tk.DISABLED)
    
    # Button click handlers
    def _on_start_click(self) -> None:
        if self.on_start_camera:
            self.running = True
            self.paused = False
            self._update_button_states()
            self.set_status_bar("Starting camera…")

            # on_start_camera() returns (success: bool, error_msg: str)
            # If the caller is wired up to return this tuple, show an error dialog.
            result = self.on_start_camera()

            if result is False or (isinstance(result, tuple) and not result[0]):
                # Camera failed to start — reset GUI state
                self.running = False
                self.paused = False
                self._update_button_states()

                # Pull error message from the tuple, or use a generic fallback
                if isinstance(result, tuple) and len(result) > 1:
                    error_msg = result[1]
                else:
                    error_msg = (
                        "No webcam detected.\n\n"
                        "Your laptop may not have a built-in camera, or the camera\n"
                        "may be disabled or in use by another application.\n\n"
                        "What you can do:\n"
                        "  1. Click 'Load Video' to use a .mp4 / .avi file.\n"
                        "  2. Connect an external USB webcam and try again.\n"
                        "  3. Click 'Connect IP Camera' to use an RTSP stream.\n"
                        "  4. Run:  python3 data/generate_test_video.py\n"
                        "     …then click 'Load Video' and select data/test_cctv.mp4"
                    )

                self.set_status_bar("⚠ Camera not available — see instructions below")
                messagebox.showwarning(
                    "Camera Not Connected",
                    error_msg,
                    parent=self.root
                )
    
    def _on_stop_click(self) -> None:
        if self.on_stop_camera:
            self.running = False
            self.paused = False
            self._update_button_states()
            self.set_status_bar("Camera stopped")
            self.on_stop_camera()
            # Clear video
            self.current_frame = None
            self.video_canvas.delete("all")
            self.status_vars["camera"].set("Disconnected")
            self.status_vars["fps"].set("0.0")
            self.status_vars["people_count"].set("0")
    
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
        if self.on_load_video:
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
            )
            if file_path:
                self.set_status_bar(f"Loading video: {Path(file_path).name}")
                self.on_load_video(file_path)
    
    def _on_connect_ip_click(self) -> None:
        if self.on_connect_ip_camera:
            # Simple dialog for RTSP URL
            dialog = tk.Toplevel(self.root)
            dialog.title("Connect IP Camera")
            dialog.geometry("400x150")
            dialog.transient(self.root)
            dialog.grab_set()
            
            ttk.Label(dialog, text="RTSP URL:").pack(pady=10)
            url_var = tk.StringVar(value="rtsp://")
            entry = ttk.Entry(dialog, textvariable=url_var, width=50)
            entry.pack(pady=5)
            entry.focus()
            
            def connect():
                url = url_var.get().strip()
                if url:
                    dialog.destroy()
                    self.set_status_bar(f"Connecting to IP camera...")
                    self.on_connect_ip_camera(url)
            
            ttk.Button(dialog, text="Connect", command=connect).pack(pady=10)
            entry.bind("<Return>", lambda e: connect())
    
    def _on_screenshot_click(self) -> None:
        if self.on_screenshot:
            self.on_screenshot()
            self.set_status_bar("Screenshot saved")
    
    def _on_export_report_click(self) -> None:
        if self.on_export_report:
            self.set_status_bar("Exporting report...")
            self.on_export_report()
    
    def _on_clear_events_click(self) -> None:
        if messagebox.askyesno("Clear Events", "Clear all events from the log?"):
            self.clear_events()
            if self.on_clear_events:
                self.on_clear_events()
            self.set_status_bar("Events cleared")
    
    def _on_settings_click(self) -> None:
        if self.on_settings:
            self.on_settings()
    
    def run(self) -> None:
        """Start the Tkinter main loop."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()
    
    def _on_closing(self) -> None:
        """Handle window close event."""
        if self.running:
            if messagebox.askokcancel("Quit", "Camera is running. Stop and quit?"):
                if self.on_stop_camera:
                    self.on_stop_camera()
                self.root.destroy()
        else:
            self.root.destroy()


def launch_gui(config_manager: ConfigManager) -> ApplicationGUI:
    """Launch the GUI application."""
    app = ApplicationGUI(config_manager)
    return app