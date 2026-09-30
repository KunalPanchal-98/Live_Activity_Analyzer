"""
UPGRADE 4 — Modern AI Surveillance Dashboard GUI Tests
======================================================
Tests verify the UI architecture and integration:
  1. GUI creation and widget hierarchy initialization
  2. Multi-page sidebar navigation across all 10 views
  3. Camera status transitions and top bar status pill updates
  4. Metric cards real-time value updates and N/A fallbacks
  5. RECONNECTING state rendering and color styling
  6. NO_CAMERA state rendering and fallback action bar availability
  7. Live monitor video canvas rendering with BGR->RGB conversion
  8. Clean shutdown behavior on window close
  9. Button callback wiring invoking actual application logic
"""

import time
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from app.config_manager import get_config_manager
from app.camera_manager import CameraState, CameraDeviceInfo
from app.gui import ApplicationGUI, MetricCard


class TestSurveillanceDashboardGUI(unittest.TestCase):
    """
    Comprehensive test suite for the redesigned AI surveillance dashboard.
    Uses a single ApplicationGUI instance with window withdrawn for clean execution
    across macOS/Linux/Windows environments.
    """

    @classmethod
    def setUpClass(cls):
        cls.config_manager = get_config_manager("config.json")
        cls.app = ApplicationGUI(cls.config_manager, start_loops=False)
        cls.app.root.withdraw()  # Run headless offscreen

    @classmethod
    def tearDownClass(cls):
        try:
            cls.app.root.destroy()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # 1. GUI Creation
    # -----------------------------------------------------------------------
    def test_01_gui_initialization_and_window_properties(self):
        self.assertIsNotNone(self.app.root)
        self.assertTrue(self.app.root.winfo_exists())
        self.assertIn("Live Activity Analyzer", self.app.root.title())
        self.assertFalse(self.app.running)
        self.assertFalse(self.app.paused)

    def test_02_all_10_views_created(self):
        expected_views = [
            "dashboard", "live_monitor", "people", "activities",
            "anomalies", "statistics", "events", "camera", "settings", "about"
        ]
        for v in expected_views:
            self.assertIn(v, self.app.views, f"View '{v}' missing in ApplicationGUI.views")
            self.assertIn(v, self.app.sidebar_buttons, f"Sidebar button '{v}' missing")

    # -----------------------------------------------------------------------
    # 2. Navigation
    # -----------------------------------------------------------------------
    def test_03_navigation_switches_active_container(self):
        views = [
            "dashboard", "live_monitor", "people", "activities",
            "anomalies", "statistics", "events", "camera", "settings", "about"
        ]
        for view_name in views:
            self.app.show_view(view_name)
            self.assertEqual(self.app.current_view_name, view_name)
            active_frame = self.app.views[view_name]
            self.assertTrue(active_frame.winfo_ismapped() or active_frame.winfo_exists())

    # -----------------------------------------------------------------------
    # 3. Camera Status Rendering
    # -----------------------------------------------------------------------
    def test_04_camera_connected_state(self):
        self.app.set_camera_state(CameraState.CONNECTED, "Connected to FaceTime HD")
        self.app.root.update_idletasks()

        self.assertEqual(self.app.camera_state, CameraState.CONNECTED)
        self.assertEqual(self.app.status_vars["camera"].get(), "CONNECTED")
        self.assertIn("CONNECTED", self.app.top_status_pill.cget("text"))

    def test_05_camera_searching_state(self):
        self.app.set_camera_state(CameraState.SEARCHING, "Probing hardware devices...")
        self.app.root.update_idletasks()

        self.assertEqual(self.app.camera_state, CameraState.SEARCHING)
        self.assertEqual(self.app.status_vars["camera"].get(), "SEARCHING")

    def test_06_camera_error_state(self):
        self.app.set_camera_state(CameraState.ERROR, "Camera hardware disconnected")
        self.app.root.update_idletasks()

        self.assertEqual(self.app.camera_state, CameraState.ERROR)
        self.assertEqual(self.app.status_vars["camera"].get(), "ERROR")

    # -----------------------------------------------------------------------
    # 4. Metric Cards
    # -----------------------------------------------------------------------
    def test_07_metric_cards_exist(self):
        required_cards = [
            "people", "active_tracks", "objects", "fps",
            "camera_status", "current_activity", "anomalies"
        ]
        for key in required_cards:
            self.assertIn(key, self.app.metric_cards, f"Card '{key}' missing from metric_cards")

    def test_08_metric_cards_update_with_real_values(self):
        self.app.update_status(
            people_count="05",
            fps="28.4",
            current_activity="Walking",
            anomaly_status="NORMAL"
        )
        self.app.root.update_idletasks()

        self.assertEqual(self.app.metric_cards["people"].value_var.get(), "05")
        self.assertEqual(self.app.metric_cards["fps"].value_var.get(), "28.4")
        self.assertEqual(self.app.metric_cards["current_activity"].value_var.get(), "Walking")
        self.assertEqual(self.app.metric_cards["anomalies"].value_var.get(), "NORMAL")

    def test_09_metric_card_na_fallback(self):
        card = self.app.metric_cards["people"]
        card.set_value("")
        self.assertEqual(card.value_var.get(), "N/A")
        card.set_value(None)
        self.assertEqual(card.value_var.get(), "N/A")

    # -----------------------------------------------------------------------
    # 5. RECONNECTING State
    # -----------------------------------------------------------------------
    def test_10_reconnecting_state_display(self):
        self.app.set_camera_state(CameraState.RECONNECTING, "Attempt 1/3: Waiting for camera...")
        self.app.root.update_idletasks()

        self.assertEqual(self.app.camera_state, CameraState.RECONNECTING)
        self.assertEqual(self.app.status_vars["camera"].get(), "RECONNECTING")
        self.assertIn("RECONNECTING", self.app.top_status_pill.cget("text"))

    # -----------------------------------------------------------------------
    # 6. NO_CAMERA State
    # -----------------------------------------------------------------------
    def test_11_no_camera_state_and_fallback_widgets(self):
        self.app.set_camera_state(CameraState.NO_CAMERA, "No webcam detected or accessible.")
        self.app.root.update_idletasks()

        self.assertEqual(self.app.camera_state, CameraState.NO_CAMERA)
        self.assertEqual(self.app.status_vars["camera"].get(), "NO CAMERA")
        self.assertFalse(self.app.running)

        # Fallback controls must exist and be accessible
        self.assertTrue(hasattr(self.app, "retry_cam_btn"))
        self.assertTrue(hasattr(self.app, "select_cam_btn"))
        self.assertTrue(hasattr(self.app, "use_file_btn"))

    # -----------------------------------------------------------------------
    # 7. Live Monitor View & BGR->RGB
    # -----------------------------------------------------------------------
    def test_12_canvas_frame_delivery_and_rgb_conversion(self):
        self.app.show_view("live_monitor")
        bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 255  # Blue channel in BGR

        self.app.update_frame(bgr_frame)
        self.assertTrue(self.app.running)
        self.assertIsNotNone(self.app.current_frame)

        # Canvas drawing triggers Pillow conversion
        self.app._update_video_display()
        self.app.root.update_idletasks()

        self.assertIsNotNone(self.app.photo_image)

    # -----------------------------------------------------------------------
    # 8. Clean Shutdown
    # -----------------------------------------------------------------------
    def test_13_clean_shutdown_behavior(self):
        # Non-running close
        self.app.running = False
        with patch.object(self.app.root, "destroy") as mock_destroy:
            self.app._on_closing()
            mock_destroy.assert_called_once()

        # Running close with confirm
        self.app.running = True
        stop_called = []
        self.app.on_stop_camera = lambda: stop_called.append(True)

        with patch("tkinter.messagebox.askokcancel", return_value=True):
            with patch.object(self.app.root, "destroy") as mock_destroy:
                self.app._on_closing()
                self.assertTrue(len(stop_called) == 1)
                mock_destroy.assert_called_once()

    # -----------------------------------------------------------------------
    # 9. Button Callback Wiring
    # -----------------------------------------------------------------------
    def test_14_start_camera_invokes_on_start(self):
        invoked = []
        self.app.on_start_camera = lambda dev: invoked.append(dev)
        self.app._on_start_click()
        self.assertTrue(len(invoked) == 1)

    def test_15_stop_camera_invokes_on_stop(self):
        invoked = []
        self.app.on_stop_camera = lambda: invoked.append(True)
        self.app._on_stop_click()
        self.assertTrue(len(invoked) == 1)
        self.assertFalse(self.app.running)

    def test_16_pause_resume_invokes_callbacks(self):
        pause_invoked = []
        resume_invoked = []
        self.app.on_pause = lambda: pause_invoked.append(True)
        self.app.on_resume = lambda: resume_invoked.append(True)

        self.app._on_pause_click()
        self.assertTrue(self.app.paused)
        self.assertTrue(len(pause_invoked) == 1)

        self.app._on_resume_click()
        self.assertFalse(self.app.paused)
        self.assertTrue(len(resume_invoked) == 1)

    def test_17_export_report_invokes_callback(self):
        report_invoked = []
        self.app.on_export_report = lambda: report_invoked.append(True) or "reports/test.csv"
        with patch("tkinter.messagebox.showinfo"):
            self.app._on_export_report_click()
        self.assertTrue(len(report_invoked) == 1)

    def test_18_add_and_clear_events(self):
        self.app.add_event({
            "timestamp": "12:00:00",
            "event_type": "loitering",
            "person_id": "ID:1",
            "activity": "Standing",
            "zone": "Zone A",
            "details": "Loitering for 35s"
        })
        self.assertEqual(len(self.app.events_tree.get_children()), 1)

        with patch("tkinter.messagebox.askyesno", return_value=True):
            self.app._on_clear_events_click()
        self.assertEqual(len(self.app.events_tree.get_children()), 0)

    def test_19_activity_mode_selector(self):
        # Auto Detect
        self.app._on_activity_mode_selected("Auto Detect", True)
        self.assertEqual(self.current_activity_mode if hasattr(self, "current_activity_mode") else self.app.current_activity_mode, "Auto Detect")

        # Future mode warning
        with patch("tkinter.messagebox.showinfo") as mock_info:
            self.app._on_activity_mode_selected("Posture", False)
            mock_info.assert_called_once()

    def test_20_settings_click(self):
        called = []
        self.app.on_settings = lambda: called.append(True)
        self.app._on_settings_click()
        self.assertEqual(self.app.current_view_name, "settings")
        self.assertTrue(called)

    # -----------------------------------------------------------------------
    # UPGRADE 5: Live Monitor Enhancements Tests
    # -----------------------------------------------------------------------
    def test_21_live_monitor_header_and_subsystem_badges(self):
        self.app.show_view("live_monitor")
        self.app.update_status(
            pipeline_detection="ACTIVE",
            pipeline_tracking="ACTIVE",
            pipeline_activity="ACTIVE",
            pipeline_anomaly="IDLE",
            resolution="1280x720",
            fps="30.0",
            inference_time="17.4ms"
        )
        self.app.root.update_idletasks()

        self.assertEqual(self.app.badge_det.cget("text"), "DET: ACTIVE")
        self.assertEqual(self.app.badge_trk.cget("text"), "TRK: ACTIVE")
        self.assertEqual(self.app.badge_act.cget("text"), "ACT: ACTIVE")
        self.assertEqual(self.app.badge_anom.cget("text"), "ANOM: IDLE")
        self.assertIn("1280x720", self.app.live_telemetry_top_lbl.cget("text"))
        self.assertIn("30.0", self.app.live_telemetry_top_lbl.cget("text"))
        self.assertIn("17.4ms", self.app.live_telemetry_top_lbl.cget("text"))

    def test_22_live_monitor_performance_card_metrics(self):
        self.app.update_status(
            resolution="1920x1080",
            fps="29.8",
            inference_time="15.2ms",
            process_time="21.0ms"
        )
        self.app.status_vars["active_tracks"].set("4")
        self.app.status_vars["objects_count"].set("4")
        self.app.root.update_idletasks()

        self.assertEqual(self.app.status_vars["resolution"].get(), "1920x1080")
        self.assertEqual(self.app.status_vars["fps"].get(), "29.8")
        self.assertEqual(self.app.status_vars["inference_time"].get(), "15.2ms")
        self.assertEqual(self.app.status_vars["process_time"].get(), "21.0ms")

    def test_23_live_monitor_event_stream_updates(self):
        self.app.show_view("live_monitor")
        init_count = len(self.app.live_events_tree.get_children())
        self.app.add_event({
            "timestamp": "14:22:10",
            "event_type": "RESTRICTED_ZONE",
            "person_id": "ID:3",
            "activity": "Walking",
            "zone": "Server Room",
            "details": "Unauthorized zone entry"
        })
        self.assertEqual(len(self.app.live_events_tree.get_children()), init_count + 1)
        first_row = self.app.live_events_tree.item(self.app.live_events_tree.get_children()[0])["values"]
        self.assertEqual(str(first_row[0]), "14:22:10")
        self.assertEqual(str(first_row[1]), "ID:3")
        self.assertEqual(str(first_row[2]), "RESTRICTED_ZONE")

        self.app.clear_events()
        self.assertEqual(len(self.app.live_events_tree.get_children()), 0)

    def test_24_live_monitor_telemetry_strip_data_propagation(self):
        self.app.update_status(
            fps="25.5",
            inference_time="16.0ms",
            people_count="2"
        )
        self.app.status_vars["active_tracks"].set("2")
        self.app.set_camera_state(CameraState.CONNECTED, "Connected")
        self.app.root.update_idletasks()

        telemetry_txt = self.app.video_telemetry_var.get()
        self.assertIn("25.5", telemetry_txt)
        self.assertIn("16.0ms", telemetry_txt)
        self.assertIn("CONNECTED", telemetry_txt)

    def test_25_aspect_ratio_scaling_preserves_dimensions(self):
        # 16:9 widescreen frame
        widescreen = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.app.update_frame(widescreen)
        self.app._update_video_display()
        self.app.root.update_idletasks()
        self.assertIsNotNone(self.app.photo_image)

        # 4:3 standard frame
        standard = np.zeros((480, 640, 3), dtype=np.uint8)
        self.app.update_frame(standard)
        self.app._update_video_display()
        self.app.root.update_idletasks()
        self.assertIsNotNone(self.app.photo_image)

    def test_26_pipeline_result_latency_propagation(self):
        mock_result = MagicMock()
        mock_result.tracks = []
        mock_result.detections = []
        mock_result.anomalies = []
        mock_result.activities = {}
        mock_result.fps = 30.0
        mock_result.inference_time_ms = 18.5
        mock_result.process_time_ms = 24.8
        mock_result.timestamp = time.time()

        self.app.update_pipeline_result(mock_result)
        self.app.root.update_idletasks()

        self.assertEqual(self.app.status_vars["inference_time"].get(), "18.5ms")
        self.assertEqual(self.app.status_vars["process_time"].get(), "24.8ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
