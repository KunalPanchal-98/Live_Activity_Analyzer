"""
Statistics Manager for Live Activity Analyzer
Uses Pandas for data analysis and Matplotlib for visualization.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta
import io
import base64

from app.config_manager import get_config
from app.database import get_database


@dataclass
class StatSnapshot:
    """Statistics snapshot at a point in time."""
    timestamp: float
    total_persons: int
    current_persons: int
    total_entries: int
    total_exits: int
    walking_count: int
    standing_count: int
    running_count: int
    loitering_events: int
    anomaly_count: int
    restricted_zone_events: int
    avg_activity_duration: float
    crowd_level: int


class StatisticsManager:
    """Manages statistics collection, analysis, and visualization."""

    def __init__(self) -> None:
        self._config = get_config()
        self._db = get_database()
        self._enabled: bool = self._config.get("statistics.enabled", True)
        self._update_interval: float = self._config.get("statistics.update_interval", 5.0)
        self._chart_path: Path = Path(self._config.get("statistics.chart_path", "reports/charts"))
        self._max_data_points: int = self._config.get("statistics.max_data_points", 1000)

        self._chart_path.mkdir(parents=True, exist_ok=True)

        self._snapshots: deque = deque(maxlen=self._max_data_points)
        self._last_update: float = 0.0
        self._session_id: Optional[int] = None
        self._lock = threading.RLock()
        self._logger = logging.getLogger("StatisticsManager")

        # Current session stats
        self._session_stats = {
            "total_detections": 0,
            "total_entries": 0,
            "total_exits": 0,
            "activity_counts": {},
            "event_counts": {},
            "zone_events": {},
            "anomaly_count": 0,
            "start_time": time.time()
        }

    def set_session(self, session_id: int) -> None:
        """Set current session."""
        self._session_id = session_id
        self._session_stats["start_time"] = time.time()

    def update(self, tracks: List, activities: Dict[int, str],
               events: List, anomalies: List, timestamp: float) -> None:
        """Update statistics with current frame data."""
        if not self._enabled:
            return

        with self._lock:
            current_time = time.time()
            if current_time - self._last_update < self._update_interval:
                return
            self._last_update = current_time

            # Count activities
            activity_counts = {}
            for act in activities.values():
                activity_counts[act] = activity_counts.get(act, 0) + 1

            # Count events by type
            event_counts = {}
            for event in events:
                if hasattr(event, 'event_type'):
                    etype = event.event_type
                elif isinstance(event, dict):
                    etype = event.get('event_type', 'unknown')
                else:
                    etype = str(type(event))
                event_counts[etype] = event_counts.get(etype, 0) + 1

            # Create snapshot
            snapshot = StatSnapshot(
                timestamp=timestamp,
                total_persons=self._session_stats["total_detections"],
                current_persons=len(tracks),
                total_entries=self._session_stats["total_entries"],
                total_exits=self._session_stats["total_exits"],
                walking_count=activity_counts.get("Walking", 0),
                standing_count=activity_counts.get("Standing", 0),
                running_count=activity_counts.get("Running", 0),
                loitering_events=event_counts.get("loitering", 0),
                anomaly_count=len(anomalies),
                restricted_zone_events=event_counts.get("restricted_zone", 0),
                avg_activity_duration=0.0,  # Would need DB query
                crowd_level=len(tracks)
            )

            self._snapshots.append(snapshot)
            self._update_session_stats(activities, events, anomalies)

    def _update_session_stats(self, activities: Dict, events: List, anomalies: List) -> None:
        """Update running session statistics."""
        for act in activities.values():
            self._session_stats["activity_counts"][act] = \
                self._session_stats["activity_counts"].get(act, 0) + 1

        for event in events:
            if hasattr(event, 'event_type'):
                etype = event.event_type
            elif isinstance(event, dict):
                etype = event.get('event_type', 'unknown')
            else:
                etype = 'unknown'
            self._session_stats["event_counts"][etype] = \
                self._session_stats["event_counts"].get(etype, 0) + 1

        self._session_stats["anomaly_count"] = len(anomalies)

    def record_detection(self) -> None:
        """Record a new person detection."""
        with self._lock:
            self._session_stats["total_detections"] += 1

    def record_entry(self) -> None:
        """Record an entry event."""
        with self._lock:
            self._session_stats["total_entries"] += 1

    def record_exit(self) -> None:
        """Record an exit event."""
        with self._lock:
            self._session_stats["total_exits"] += 1

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        with self._lock:
            return {
                "session_id": self._session_id,
                "session_duration": time.time() - self._session_stats["start_time"],
                "total_detections": self._session_stats["total_detections"],
                "current_people": self._session_stats["activity_counts"].get("current", 0),
                "total_entries": self._session_stats["total_entries"],
                "total_exits": self._session_stats["total_exits"],
                "activity_distribution": self._session_stats["activity_counts"].copy(),
                "event_distribution": self._session_stats["event_counts"].copy(),
                "anomaly_count": self._session_stats["anomaly_count"],
                "snapshots_collected": len(self._snapshots)
            }

    def get_dataframe(self) -> pd.DataFrame:
        """Get snapshots as Pandas DataFrame."""
        with self._lock:
            if not self._snapshots:
                return pd.DataFrame()

            data = []
            for s in self._snapshots:
                data.append({
                    'timestamp': datetime.fromtimestamp(s.timestamp),
                    'total_persons': s.total_persons,
                    'current_persons': s.current_persons,
                    'total_entries': s.total_entries,
                    'total_exits': s.total_exits,
                    'walking': s.walking_count,
                    'standing': s.standing_count,
                    'running': s.running_count,
                    'loitering_events': s.loitering_events,
                    'anomaly_count': s.anomaly_count,
                    'restricted_zone_events': s.restricted_zone_events,
                    'crowd_level': s.crowd_level
                })
            return pd.DataFrame(data)

    def generate_activity_distribution_chart(self) -> Optional[str]:
        """Generate activity distribution pie chart."""
        with self._lock:
            df = self.get_dataframe()
            if df.empty:
                return None

            # Get latest activity counts
            latest = df.iloc[-1]
            activities = ['walking', 'standing', 'running']
            counts = [latest['walking'], latest['standing'], latest['running']]
            colors = ['#3498db', '#2ecc71', '#e74c3c']

            fig, ax = plt.subplots(figsize=(6, 6))
            if sum(counts) == 0:
                ax.pie([1], labels=['No Activity Recorded'], colors=['#bdc3c7'], autopct='%1.0f%%')
            else:
                wedges, texts, autotexts = ax.pie(
                    counts, labels=activities, colors=colors,
                    autopct='%1.1f%%', startangle=90
                )
            ax.set_title('Activity Distribution', fontsize=14, fontweight='bold')

            return self._fig_to_base64(fig)

    def generate_people_over_time_chart(self) -> Optional[str]:
        """Generate people detected over time line chart."""
        with self._lock:
            df = self.get_dataframe()
            if df.empty or len(df) < 2:
                return None

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['timestamp'], df['current_persons'], 'b-', linewidth=2, label='Current People')
            ax.plot(df['timestamp'], df['total_persons'], 'g--', linewidth=1, label='Total Detections')
            ax.fill_between(df['timestamp'], df['current_persons'], alpha=0.3, color='blue')

            ax.set_xlabel('Time')
            ax.set_ylabel('Number of People')
            ax.set_title('People Detected Over Time', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()

            return self._fig_to_base64(fig)

    def generate_anomalies_over_time_chart(self) -> Optional[str]:
        """Generate anomalies over time chart."""
        with self._lock:
            df = self.get_dataframe()
            if df.empty:
                return None

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df['timestamp'], df['anomaly_count'], color='#e74c3c', alpha=0.7, width=0.001)
            ax.set_xlabel('Time')
            ax.set_ylabel('Anomaly Count')
            ax.set_title('Anomalies Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()

            return self._fig_to_base64(fig)

    def generate_hourly_activity_chart(self) -> Optional[str]:
        """Generate hourly activity chart."""
        with self._lock:
            if not self._session_id:
                return None

            # Query database for hourly stats
            try:
                events = self._db.get_events(self._session_id, limit=1000)
                if not events:
                    return None

                hours = []
                for e in events:
                    ts = e['timestamp']
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    hours.append(ts.hour)

                if not hours:
                    return None

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(hours, bins=24, range=(0, 24), color='#3498db', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Event Count')
                ax.set_title('Hourly Activity Distribution', fontsize=14, fontweight='bold')
                ax.set_xticks(range(0, 24, 2))
                ax.grid(True, alpha=0.3, axis='y')

                return self._fig_to_base64(fig)
            except Exception as e:
                self._logger.error(f"Hourly chart error: {e}")
                return None

    def generate_event_frequency_chart(self) -> Optional[str]:
        """Generate event frequency bar chart."""
        with self._lock:
            stats = self.get_current_stats()
            events = stats.get('event_distribution', {})
            if not events:
                return None

            fig, ax = plt.subplots(figsize=(10, 5))
            event_types = list(events.keys())
            counts = list(events.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(event_types)))

            bars = ax.bar(event_types, counts, color=colors, edgecolor='black')
            ax.set_xlabel('Event Type')
            ax.set_ylabel('Frequency')
            ax.set_title('Event Frequency', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')

            plt.tight_layout()
            return self._fig_to_base64(fig)

    def generate_all_charts(self) -> Dict[str, Optional[str]]:
        """Generate all charts and return as base64 strings."""
        return {
            "activity_distribution": self.generate_activity_distribution_chart(),
            "people_over_time": self.generate_people_over_time_chart(),
            "anomalies_over_time": self.generate_anomalies_over_time_chart(),
            "hourly_activity": self.generate_hourly_activity_chart(),
            "event_frequency": self.generate_event_frequency_chart()
        }

    def save_charts_to_files(self) -> Dict[str, str]:
        """Save all charts to image files."""
        saved = {}
        charts = self.generate_all_charts()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, b64_data in charts.items():
            if b64_data:
                path = self._chart_path / f"{name}_{timestamp}.png"
                try:
                    import base64
                    with open(path, 'wb') as f:
                        f.write(base64.b64decode(b64_data))
                    saved[name] = str(path)
                except Exception as e:
                    self._logger.error(f"Failed to save chart {name}: {e}")

        return saved

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive summary report."""
        with self._lock:
            df = self.get_dataframe()
            stats = self.get_current_stats()

            report = {
                "session_info": {
                    "session_id": self._session_id,
                    "duration_seconds": time.time() - self._session_stats["start_time"],
                    "snapshots": len(self._snapshots)
                },
                "detection_stats": {
                    "total_detections": stats["total_detections"],
                    "current_people": stats["current_people"],
                    "total_entries": stats["total_entries"],
                    "total_exits": stats["total_exits"]
                },
                "activity_stats": {
                    "distribution": stats["activity_distribution"],
                    "avg_duration": 0.0  # Would need DB
                },
                "event_stats": {
                    "distribution": stats["event_distribution"],
                    "total_events": sum(stats["event_distribution"].values()),
                    "anomaly_count": stats["anomaly_count"]
                }
            }

            if not df.empty:
                report["time_series"] = {
                    "avg_people": df['current_persons'].mean(),
                    "max_people": df['current_persons'].max(),
                    "total_anomalies": df['anomaly_count'].sum(),
                    "total_loitering": df['loitering_events'].sum()
                }

            return report

    def reset(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._snapshots.clear()
            self._session_stats = {
                "total_detections": 0,
                "total_entries": 0,
                "total_exits": 0,
                "activity_counts": {},
                "event_counts": {},
                "zone_events": {},
                "anomaly_count": 0,
                "start_time": time.time()
            }
            self._last_update = 0.0