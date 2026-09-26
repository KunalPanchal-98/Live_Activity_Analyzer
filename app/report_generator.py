"""
Report Generator for Live Activity Analyzer
Exports reports in CSV, Excel, and chart formats.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from app.config_manager import get_config
from app.database import get_database
from app.statistics import StatisticsManager


class ReportGenerator:
    """Generates and exports analysis reports."""

    def __init__(self) -> None:
        self._config = get_config()
        self._db = get_database()
        self._stats = StatisticsManager()
        self._enabled: bool = self._config.get("reporting.enabled", True)
        self._export_format: str = self._config.get("reporting.export_format", "csv")
        self._include_charts: bool = self._config.get("reporting.include_charts", True)
        self._reports_path: Path = Path("reports")
        self._reports_path.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("ReportGenerator")

    def generate_session_report(self, session_id: int,
                                output_path: Optional[str] = None) -> Optional[str]:
        """Generate comprehensive report for a session."""
        if not self._enabled:
            return None

        # Get session summary from database
        summary = self._db.get_session_summary(session_id)
        if not summary or summary.get("total_persons", 0) == 0:
            self._logger.warning(f"No data for session {session_id}")
            return None

        # Get detailed data
        events = self._db.get_events(session_id, limit=10000)
        persons = self._db.get_active_persons(session_id)  # Actually gets all persons for session
        activity_logs = self._db.execute_query(
            "SELECT * FROM activity_logs WHERE session_id = ? ORDER BY start_time", (session_id,)
        )

        # Prepare report data
        report_data = {
            "session_info": {
                "session_id": session_id,
                "generated_at": datetime.now().isoformat(),
                "total_persons": summary["total_persons"],
                "total_events": summary["total_events"],
                "event_types": summary["event_types"],
                "activities": summary["activities"],
                "avg_activity_duration": summary["avg_activity_duration"]
            },
            "events": [dict(e) for e in events],
            "persons": [dict(p) for p in persons],
            "activity_logs": [dict(a) for a in activity_logs]
        }

        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self._reports_path / f"session_{session_id}_report_{timestamp}")

        # Export based on format
        if self._export_format.lower() == "csv":
            return self._export_csv(report_data, output_path)
        elif self._export_format.lower() == "excel":
            return self._export_excel(report_data, output_path)
        elif self._export_format.lower() == "json":
            return self._export_json(report_data, output_path)
        else:
            self._logger.error(f"Unsupported export format: {self._export_format}")
            return None

    def _export_csv(self, data: Dict[str, Any], base_path: str) -> str:
        """Export report as CSV files."""
        base = Path(base_path)
        base.parent.mkdir(parents=True, exist_ok=True)

        # Session info CSV
        session_path = base.with_suffix('.session.csv')
        with open(session_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Field", "Value"])
            for key, value in data["session_info"].items():
                writer.writerow([key, value])

        # Events CSV
        events_path = base.with_suffix('.events.csv')
        if data["events"]:
            df = pd.DataFrame(data["events"])
            df.to_csv(events_path, index=False)

        # Persons CSV
        persons_path = base.with_suffix('.persons.csv')
        if data["persons"]:
            df = pd.DataFrame(data["persons"])
            df.to_csv(persons_path, index=False)

        # Activity logs CSV
        activities_path = base.with_suffix('.activities.csv')
        if data["activity_logs"]:
            df = pd.DataFrame(data["activity_logs"])
            df.to_csv(activities_path, index=False)

        self._logger.info(f"CSV report saved to {base.parent}")
        return str(base.parent)

    def _export_excel(self, data: Dict[str, Any], base_path: str) -> str:
        """Export report as Excel workbook."""
        try:
            base = Path(base_path)
            excel_path = base.with_suffix('.xlsx')
            base.parent.mkdir(parents=True, exist_ok=True)

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Session info sheet
                session_df = pd.DataFrame(list(data["session_info"].items()), columns=["Field", "Value"])
                session_df.to_excel(writer, sheet_name="Session Info", index=False)

                # Events sheet
                if data["events"]:
                    events_df = pd.DataFrame(data["events"])
                    events_df.to_excel(writer, sheet_name="Events", index=False)

                # Persons sheet
                if data["persons"]:
                    persons_df = pd.DataFrame(data["persons"])
                    persons_df.to_excel(writer, sheet_name="Persons", index=False)

                # Activity logs sheet
                if data["activity_logs"]:
                    acts_df = pd.DataFrame(data["activity_logs"])
                    acts_df.to_excel(writer, sheet_name="Activity Logs", index=False)

            self._logger.info(f"Excel report saved to {excel_path}")
            return str(excel_path)

        except ImportError:
            self._logger.warning("openpyxl not installed, falling back to CSV")
            return self._export_csv(data, base_path)
        except Exception as e:
            self._logger.error(f"Excel export failed: {e}")
            return self._export_csv(data, base_path)

    def _export_json(self, data: Dict[str, Any], base_path: str) -> str:
        """Export report as JSON."""
        base = Path(base_path)
        json_path = base.with_suffix('.json')
        base.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self._logger.info(f"JSON report saved to {json_path}")
        return str(json_path)

    def generate_daily_summary(self, date: Optional[datetime] = None) -> Optional[str]:
        """Generate daily summary report across all sessions."""
        if date is None:
            date = datetime.now()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Query sessions for the day
        sessions = self._db.execute_query(
            "SELECT * FROM sessions WHERE start_time BETWEEN ? AND ?",
            (start_of_day.strftime("%Y-%m-%d %H:%M:%S"), end_of_day.strftime("%Y-%m-%d %H:%M:%S"))
        )

        if not sessions:
            self._logger.info(f"No sessions found for {date.date()}")
            return None

        # Aggregate data
        daily_stats = {
            "date": date.date().isoformat(),
            "total_sessions": len(sessions),
            "total_persons": 0,
            "total_events": 0,
            "total_entries": 0,
            "total_exits": 0,
            "event_types": {},
            "activities": {},
            "anomaly_count": 0,
            "session_details": []
        }

        for session in sessions:
            sid = session["session_id"]
            summary = self._db.get_session_summary(sid)

            daily_stats["total_persons"] += summary.get("total_persons", 0)
            daily_stats["total_events"] += summary.get("total_events", 0)

            for etype, count in summary.get("event_types", {}).items():
                daily_stats["event_types"][etype] = daily_stats["event_types"].get(etype, 0) + count

            for act, count in summary.get("activities", {}).items():
                daily_stats["activities"][act] = daily_stats["activities"].get(act, 0) + count

            daily_stats["anomaly_count"] += summary.get("event_types", {}).get("anomaly", 0)

            daily_stats["session_details"].append({
                "session_id": sid,
                "start_time": session["start_time"],
                "end_time": session["end_time"],
                "camera_source": session["camera_source"],
                "persons": summary.get("total_persons", 0),
                "events": summary.get("total_events", 0)
            })

        # Save daily summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self._reports_path / f"daily_summary_{date.date()}_{timestamp}"

        if self._export_format.lower() == "csv":
            return self._export_daily_csv(daily_stats, output_path)
        elif self._export_format.lower() == "excel":
            return self._export_daily_excel(daily_stats, output_path)
        else:
            return self._export_json(daily_stats, output_path)

    def _export_daily_csv(self, data: Dict[str, Any], base_path: Path) -> str:
        """Export daily summary as CSV."""
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Main summary
        summary_path = base_path.with_suffix('.summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in data.items():
                if key != "session_details":
                    writer.writerow([key, value])

        # Session details
        sessions_path = base_path.with_suffix('.sessions.csv')
        if data["session_details"]:
            df = pd.DataFrame(data["session_details"])
            df.to_csv(sessions_path, index=False)

        return str(base_path.parent)

    def _export_daily_excel(self, data: Dict[str, Any], base_path: Path) -> str:
        """Export daily summary as Excel."""
        try:
            excel_path = base_path.with_suffix('.xlsx')
            base_path.parent.mkdir(parents=True, exist_ok=True)

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {k: v for k, v in data.items() if k != "session_details"}
                summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
                summary_df.to_excel(writer, sheet_name="Daily Summary", index=False)

                # Sessions sheet
                if data["session_details"]:
                    sessions_df = pd.DataFrame(data["session_details"])
                    sessions_df.to_excel(writer, sheet_name="Sessions", index=False)

            return str(excel_path)
        except Exception as e:
            self._logger.error(f"Daily Excel export failed: {e}")
            return self._export_daily_csv(data, base_path)

    def export_charts(self, session_id: int) -> Dict[str, str]:
        """Export charts for a session."""
        self._stats.set_session(session_id)
        return self._stats.save_charts_to_files()

    def get_report_list(self) -> List[Dict[str, Any]]:
        """Get list of generated reports."""
        reports = []
        for path in self._reports_path.glob("*"):
            if path.is_file():
                stat = path.stat()
                reports.append({
                    "name": path.name,
                    "path": str(path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        return sorted(reports, key=lambda x: x["modified"], reverse=True)

    def delete_report(self, report_name: str) -> bool:
        """Delete a report file."""
        path = self._reports_path / report_name
        if path.exists() and path.is_file():
            try:
                path.unlink()
                return True
            except Exception as e:
                self._logger.error(f"Failed to delete report: {e}")
        return False

    def set_export_format(self, fmt: str) -> None:
        """Set export format."""
        self._export_format = fmt
        self._config.set("reporting.export_format", fmt)

    def set_include_charts(self, include: bool) -> None:
        """Set whether to include charts in reports."""
        self._include_charts = include
        self._config.set("reporting.include_charts", include)