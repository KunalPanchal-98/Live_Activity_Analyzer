"""
Database Manager for Live Activity Analyzer
Handles SQLite database operations for persons, events, activity logs, sessions, and statistics.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
import json

from app.config_manager import get_config


class DatabaseManager:
    """Manages SQLite database for storing surveillance data."""

    _instance: Optional["DatabaseManager"] = None

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._config = get_config()
            self._db_path = Path(self._config.get("database.path", "database/activity.db"))
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_database()
            self._initialized = True

    def _init_database(self) -> None:
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    camera_source TEXT,
                    resolution TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    first_seen TIMESTAMP NOT NULL,
                    last_seen TIMESTAMP,
                    total_frames INTEGER DEFAULT 0,
                    total_distance REAL DEFAULT 0.0,
                    avg_speed REAL DEFAULT 0.0,
                    max_speed REAL DEFAULT 0.0,
                    activities TEXT,
                    zones_visited TEXT,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    person_id INTEGER,
                    track_id INTEGER,
                    event_type TEXT NOT NULL,
                    event_subtype TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    zone_name TEXT,
                    confidence REAL,
                    duration REAL,
                    details TEXT,
                    screenshot_path TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                    FOREIGN KEY (person_id) REFERENCES persons (person_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    activity TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration REAL,
                    zone_name TEXT,
                    confidence REAL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                    FOREIGN KEY (person_id) REFERENCES persons (person_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS zones (
                    zone_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    zone_name TEXT NOT NULL,
                    zone_type TEXT NOT NULL,
                    coordinates TEXT NOT NULL,
                    color TEXT,
                    active INTEGER DEFAULT 1,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_session_time
                ON events (session_id, timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_activity_logs_session_person
                ON activity_logs (session_id, person_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_persons_session
                ON persons (session_id)
            """)

            conn.commit()
            logging.info("Database initialized successfully")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def execute_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_update(self, query: str, params: Tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid or cursor.rowcount

    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute multiple INSERT/UPDATE/DELETE queries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount

    def create_session(self, camera_source: str, resolution: str) -> int:
        """Create a new surveillance session."""
        query = """
            INSERT INTO sessions (start_time, camera_source, resolution, status)
            VALUES (?, ?, ?, 'active')
        """
        return self.execute_update(query, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera_source, resolution))

    def end_session(self, session_id: int) -> bool:
        """End a surveillance session."""
        query = "UPDATE sessions SET end_time = ?, status = 'completed' WHERE session_id = ?"
        return self.execute_update(query, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id)) > 0

    def get_active_session(self) -> Optional[sqlite3.Row]:
        """Get the currently active session."""
        query = "SELECT * FROM sessions WHERE status = 'active' ORDER BY start_time DESC LIMIT 1"
        results = self.execute_query(query)
        return results[0] if results else None

    def add_person(self, session_id: int, track_id: int) -> int:
        """Add a new tracked person."""
        query = """
            INSERT INTO persons (session_id, track_id, first_seen, total_frames, activities, zones_visited)
            VALUES (?, ?, ?, 0, '[]', '[]')
        """
        return self.execute_update(query, (session_id, track_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def update_person(self, person_id: int, **kwargs) -> bool:
        """Update person information."""
        if not kwargs:
            return False
        fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
        query = f"UPDATE persons SET {fields} WHERE person_id = ?"
        params = tuple(kwargs.values()) + (person_id,)
        return self.execute_update(query, params) > 0

    def get_person_by_track_id(self, session_id: int, track_id: int) -> Optional[sqlite3.Row]:
        """Get person by track ID in a session."""
        query = "SELECT * FROM persons WHERE session_id = ? AND track_id = ?"
        results = self.execute_query(query, (session_id, track_id))
        return results[0] if results else None

    def get_active_persons(self, session_id: int) -> List[sqlite3.Row]:
        """Get all active persons in a session."""
        query = "SELECT * FROM persons WHERE session_id = ? AND status = 'active'"
        return self.execute_query(query, (session_id,))

    def add_event(self, session_id: int, event_type: str, **kwargs) -> int:
        """Add a new event."""
        fields = ["session_id", "event_type"]
        values = [session_id, event_type]
        for key, value in kwargs.items():
            fields.append(key)
            values.append(value)
        fields.append("timestamp")
        values.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        query = f"INSERT INTO events ({', '.join(fields)}) VALUES ({', '.join(['?'] * len(fields))})"
        return self.execute_update(query, tuple(values))

    def get_events(self, session_id: int, event_type: Optional[str] = None,
                   limit: int = 100) -> List[sqlite3.Row]:
        """Get events for a session."""
        if event_type:
            query = "SELECT * FROM events WHERE session_id = ? AND event_type = ? ORDER BY timestamp DESC LIMIT ?"
            return self.execute_query(query, (session_id, event_type, limit))
        query = "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?"
        return self.execute_query(query, (session_id, limit))

    def get_recent_events(self, session_id: int, minutes: int = 5) -> List[sqlite3.Row]:
        """Get events from the last N minutes."""
        query = """
            SELECT * FROM events
            WHERE session_id = ? AND timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """
        return self.execute_query(query, (session_id, f"-{minutes} minutes"))

    def add_activity_log(self, session_id: int, person_id: int, track_id: int,
                         activity: str, **kwargs) -> int:
        """Add an activity log entry."""
        fields = ["session_id", "person_id", "track_id", "activity", "start_time"]
        values = [session_id, person_id, track_id, activity, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        for key, value in kwargs.items():
            fields.append(key)
            values.append(value)

        query = f"INSERT INTO activity_logs ({', '.join(fields)}) VALUES ({', '.join(['?'] * len(fields))})"
        return self.execute_update(query, tuple(values))

    def update_activity_log(self, log_id: int, **kwargs) -> bool:
        """Update an activity log entry."""
        if not kwargs:
            return False
        fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
        query = f"UPDATE activity_logs SET {fields} WHERE log_id = ?"
        params = tuple(kwargs.values()) + (log_id,)
        return self.execute_update(query, params) > 0

    def get_activity_logs(self, session_id: int, person_id: Optional[int] = None,
                          activity: Optional[str] = None) -> List[sqlite3.Row]:
        """Get activity logs for a session."""
        query = "SELECT * FROM activity_logs WHERE session_id = ?"
        params = [session_id]
        if person_id:
            query += " AND person_id = ?"
            params.append(person_id)
        if activity:
            query += " AND activity = ?"
            params.append(activity)
        query += " ORDER BY start_time DESC"
        return self.execute_query(query, tuple(params))

    def add_statistic(self, session_id: int, metric_name: str, metric_value: float,
                      metric_unit: Optional[str] = None) -> int:
        """Add a statistic entry."""
        query = """
            INSERT INTO statistics (session_id, timestamp, metric_name, metric_value, metric_unit)
            VALUES (?, ?, ?, ?, ?)
        """
        return self.execute_update(query, (session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), metric_name, metric_value, metric_unit))

    def get_statistics(self, session_id: int, metric_name: Optional[str] = None,
                       limit: int = 1000) -> List[sqlite3.Row]:
        """Get statistics for a session."""
        if metric_name:
            query = """
                SELECT * FROM statistics
                WHERE session_id = ? AND metric_name = ?
                ORDER BY timestamp DESC LIMIT ?
            """
            return self.execute_query(query, (session_id, metric_name, limit))
        query = "SELECT * FROM statistics WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?"
        return self.execute_query(query, (session_id, limit))

    def add_zone(self, session_id: int, zone_name: str, zone_type: str,
                 coordinates: List[Tuple[float, float]], color: str = "#FF0000") -> int:
        """Add a zone definition."""
        query = """
            INSERT INTO zones (session_id, zone_name, zone_type, coordinates, color)
            VALUES (?, ?, ?, ?, ?)
        """
        return self.execute_update(query, (session_id, zone_name, zone_type,
                                           json.dumps(coordinates), color))

    def get_zones(self, session_id: int, zone_type: Optional[str] = None) -> List[sqlite3.Row]:
        """Get zones for a session."""
        if zone_type:
            query = "SELECT * FROM zones WHERE session_id = ? AND zone_type = ? AND active = 1"
            return self.execute_query(query, (session_id, zone_type))
        query = "SELECT * FROM zones WHERE session_id = ? AND active = 1"
        return self.execute_query(query, (session_id,))

    def get_session_summary(self, session_id: int) -> Dict[str, Any]:
        """Get summary statistics for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM persons WHERE session_id = ?", (session_id,))
            total_persons = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM events WHERE session_id = ?", (session_id,))
            total_events = cursor.fetchone()["count"]

            cursor.execute("SELECT event_type, COUNT(*) as count FROM events WHERE session_id = ? GROUP BY event_type",
                           (session_id,))
            event_types = {row["event_type"]: row["count"] for row in cursor.fetchall()}

            cursor.execute("SELECT activity, COUNT(*) as count FROM activity_logs WHERE session_id = ? GROUP BY activity",
                           (session_id,))
            activities = {row["activity"]: row["count"] for row in cursor.fetchall()}

            cursor.execute("SELECT AVG(duration) as avg_duration FROM activity_logs WHERE session_id = ? AND duration IS NOT NULL",
                           (session_id,))
            avg_duration = cursor.fetchone()["avg_duration"] or 0.0

            return {
                "session_id": session_id,
                "total_persons": total_persons,
                "total_events": total_events,
                "event_types": event_types,
                "activities": activities,
                "avg_activity_duration": avg_duration
            }

    def cleanup_old_data(self, days: int = 30) -> int:
        """Remove data older than specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE start_time < datetime('now', ?) AND status = 'completed'",
                           (f"-{days} days",))
            deleted = cursor.rowcount
            conn.commit()
            return deleted


def get_database() -> DatabaseManager:
    """Get the global DatabaseManager instance."""
    return DatabaseManager()