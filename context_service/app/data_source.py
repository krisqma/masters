"""Reads IoT sensor data from CSV and produces human-readable snapshots for the Teacher.

Simulation mode: the CSV contains ~88 days of data at 5-min intervals (~25 200 rows).
SensorSimulator loads the entire file once, then advances a cursor on each call,
simulating real-time progression through the dataset.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Column name structure: 56_{room}_{device}_{number}_{sensor_type}_{metric}
# Two-word rooms: large_room, small_room. Others: bathroom, hallway, outside, localization.

KNOWN_ROOMS = {"bathroom", "large_room", "small_room", "hallway", "outside", "localization"}

ROOM_LABELS: Dict[str, str] = {
    "bathroom": "Łazienka",
    "large_room": "Duży pokój",
    "small_room": "Mały pokój",
    "hallway": "Korytarz",
    "outside": "Na zewnątrz",
    "localization": "Lokalizacja",
}

INTERESTING_METRICS = {
    "temperature", "humidity", "pressure", "power", "energy",
    "voc", "occupancy", "contact", "illuminance", "state",
}

SKIP_METRICS = {
    "voltage", "linkquality", "battery", "device_temperature",
    "energy_returned", "power_reactive", "pf", "trigger_count",
}


@dataclass
class SensorSnapshot:
    timestamp: str
    readings: Dict[str, Dict[str, str]]  # room -> {label: value}
    row_count: int
    cursor: int  # current position in the CSV
    total_rows: int

    def to_text(self) -> str:
        if not self.readings:
            return f"[{self.timestamp}] Brak odczytów z czujników."

        lines = [f"Stan czujników na {self.timestamp} (wiersz {self.cursor}/{self.total_rows}):"]
        for room_key, sensors in sorted(self.readings.items()):
            room_name = ROOM_LABELS.get(room_key, room_key)
            lines.append(f"\n  {room_name}:")
            for label, value in sorted(sensors.items()):
                lines.append(f"    - {label}: {value}")
        return "\n".join(lines)


def _parse_column(col: str) -> Optional[tuple[str, str, str]]:
    if col == "_time":
        return None

    parts = col.split("_")
    if len(parts) < 4 or parts[0] != "56":
        return None

    metric = parts[-1]
    if metric in SKIP_METRICS:
        return None
    if metric not in INTERESTING_METRICS:
        return None

    two_word = f"{parts[1]}_{parts[2]}" if len(parts) > 2 else ""
    if two_word in KNOWN_ROOMS:
        room = two_word
        rest_start = 3
    elif parts[1] in KNOWN_ROOMS:
        room = parts[1]
        rest_start = 2
    else:
        return None

    device_parts = parts[rest_start:-1]
    device_desc = " ".join(device_parts)
    return room, device_desc, metric


METRIC_UNITS: Dict[str, str] = {
    "temperature": "°C",
    "humidity": "%",
    "pressure": "hPa",
    "power": "W",
    "energy": "Wh",
    "voc": "ppb",
    "illuminance": "lx",
}

METRIC_LABELS: Dict[str, str] = {
    "temperature": "temperatura",
    "humidity": "wilgotność",
    "pressure": "ciśnienie",
    "power": "moc",
    "energy": "energia",
    "voc": "VOC",
    "occupancy": "obecność",
    "contact": "kontakt",
    "illuminance": "oświetlenie",
    "state": "stan",
}


def _format_value(metric: str, raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""

    unit = METRIC_UNITS.get(metric, "")

    if metric == "occupancy":
        return "tak" if raw in ("1", "1.0", "true", "True") else "nie"
    if metric == "contact":
        return "zamknięty" if raw in ("1", "1.0", "true", "True") else "otwarty"

    try:
        num = float(raw)
        if num == int(num) and metric not in ("temperature", "pressure"):
            return f"{int(num)}{' ' + unit if unit else ''}"
        return f"{num:.1f}{' ' + unit if unit else ''}"
    except ValueError:
        return f"{raw}{' ' + unit if unit else ''}"


def _build_snapshot(
    header: List[str],
    rows: List[List[str]],
    col_info: List[Optional[tuple[str, str, str]]],
    cursor: int,
    total_rows: int,
    lookback: int = 200,
) -> SensorSnapshot:
    """Build a snapshot from a window of rows ending at `cursor`."""
    window_start = max(0, cursor - lookback)
    window = rows[window_start:cursor]

    if not window:
        return SensorSnapshot(timestamp="?", readings={}, row_count=0, cursor=cursor, total_rows=total_rows)

    timestamp = window[-1][0] if window[-1][0] else "?"

    seen: set[tuple[str, str]] = set()
    readings: Dict[str, Dict[str, str]] = {}

    for row in reversed(window):
        for i, cell in enumerate(row):
            if i >= len(col_info) or col_info[i] is None:
                continue
            cell = cell.strip()
            if not cell:
                continue

            room, device_desc, metric = col_info[i]
            key = (room, f"{device_desc}_{metric}")
            if key in seen:
                continue
            seen.add(key)

            formatted = _format_value(metric, cell)
            if not formatted:
                continue

            label_metric = METRIC_LABELS.get(metric, metric)
            label = f"{device_desc} – {label_metric}"
            readings.setdefault(room, {})[label] = formatted

    return SensorSnapshot(
        timestamp=timestamp,
        readings=readings,
        row_count=len(window),
        cursor=cursor,
        total_rows=total_rows,
    )


class SensorSimulator:
    """Simulates real-time sensor data by advancing through a CSV file.

    Each call to `next_snapshot()` moves the cursor forward by `step` rows
    (default 1 = 5 minutes of simulated time per refresh cycle).
    When the cursor reaches the end, it wraps around to the beginning.
    """

    def __init__(self, csv_path: Path, step: int = 1, lookback: int = 200) -> None:
        self._csv_path = csv_path
        self._step = step
        self._lookback = lookback

        self._header: List[str] = []
        self._rows: List[List[str]] = []
        self._col_info: List[Optional[tuple[str, str, str]]] = []
        self._cursor: int = 0
        self._loaded = False

    def _load(self) -> None:
        """Load CSV into memory (once)."""
        if self._loaded:
            return

        path = self._csv_path
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[1] / path).resolve()

        if not path.exists():
            logger.error("CSV not found: %s", path)
            self._loaded = True
            return

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            self._header = next(reader)
            self._rows = list(reader)

        self._col_info = [_parse_column(h) for h in self._header]
        # Start cursor at a point where lookback has enough data
        self._cursor = min(self._lookback, len(self._rows))
        self._loaded = True
        logger.info("CSV loaded: %d rows, cursor starts at %d", len(self._rows), self._cursor)

    def next_snapshot(self) -> SensorSnapshot:
        """Advance cursor and return snapshot at current position."""
        self._load()

        if not self._rows:
            return SensorSnapshot(timestamp="?", readings={}, row_count=0, cursor=0, total_rows=0)

        snapshot = _build_snapshot(
            self._header,
            self._rows,
            self._col_info,
            cursor=self._cursor,
            total_rows=len(self._rows),
            lookback=self._lookback,
        )

        # Advance cursor
        self._cursor += self._step
        if self._cursor > len(self._rows):
            self._cursor = min(self._lookback, len(self._rows))
            logger.info("CSV simulation wrapped around to the beginning.")

        return snapshot
