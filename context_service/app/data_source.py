"""Reads IoT sensor data from CSV and produces point-in-time snapshots for the Teacher.

Simulation mode: the CSV contains ~88 days of data at 5-min intervals (~25 200 rows).
SensorSimulator loads the entire file once, then advances a cursor on each call,
simulating real-time progression through the dataset one timestamp at a time.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "localization": "Instalacja ogólna",
}

# Rooms that make up the apartment (exclude outdoor + whole-house meters).
APARTMENT_ROOM_KEYS = ("bathroom", "hallway", "large_room", "small_room")

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
    apartment_rooms: Optional[List[str]] = None
    # Structured readings (room_key, device_desc, metric, formatted) for the compact Student format.
    entries: List[Tuple[str, str, str, str]] = field(default_factory=list)

    def to_text(self) -> str:
        """Full snapshot text for the Teacher (grouped by room)."""
        if not self.readings:
            return f"[{self.timestamp}] Brak odczytów z czujników."

        lines = [
            f"Snapshot czujników z {self.timestamp} (wiersz {self.cursor}/{self.total_rows}):",
        ]
        if self.apartment_rooms:
            lines.append(
                "Pomieszczenia mieszkania (pełna lista): "
                + ", ".join(self.apartment_rooms)
                + "."
            )
        for room_key, sensors in sorted(self.readings.items()):
            room_name = ROOM_LABELS.get(room_key, room_key)
            lines.append(f"\n  {room_name}:")
            for label, value in sorted(sensors.items()):
                lines.append(f"    - {label}: {value}")
        return "\n".join(lines)

    def to_student_facts_text(self) -> str:
        """Compact per-room facts for the Student (token budget, see render_student_facts)."""
        return render_student_facts(self)[0]

    def to_student_facts_text_verbose(self) -> str:
        """Legacy bullet list (one line per reading), kept for tests and A/B comparison."""
        if not self.readings:
            return f"Stan mieszkania na {self.timestamp} UTC:\n(brak odczytów z czujników)"

        lines = [f"Stan mieszkania na {self.timestamp} UTC:"]
        rooms = self.apartment_rooms or []
        if rooms:
            lines.append(f"Pomieszczenia mieszkania (pełna lista): {', '.join(rooms)}.")
        for room_key, sensors in sorted(self.readings.items()):
            room_name = ROOM_LABELS.get(room_key, room_key)
            for label, value in sorted(sensors.items()):
                lines.append(f"- {room_name}: {label} = {value}")
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
    row: List[str],
    col_info: List[Optional[tuple[str, str, str]]],
    row_number: int,
    total_rows: int,
) -> SensorSnapshot:
    """Build a snapshot from a single CSV row."""
    if not row:
        return SensorSnapshot(timestamp="?", readings={}, row_count=0, cursor=row_number, total_rows=total_rows)

    timestamp = row[0] if row[0] else "?"
    readings: Dict[str, Dict[str, str]] = {}
    entries: List[Tuple[str, str, str, str]] = []

    for i, cell in enumerate(row):
        if i >= len(col_info) or col_info[i] is None:
            continue
        cell = cell.strip()
        if not cell:
            continue

        room, device_desc, metric = col_info[i]
        formatted = _format_value(metric, cell)
        if not formatted:
            continue

        label_metric = METRIC_LABELS.get(metric, metric)
        label = f"{device_desc} – {label_metric}"
        readings.setdefault(room, {})[label] = formatted
        entries.append((room, device_desc, metric, formatted))

    return SensorSnapshot(
        timestamp=timestamp,
        readings=readings,
        row_count=1,
        cursor=row_number,
        total_rows=total_rows,
        apartment_rooms=None,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Compact Student facts
#
# The Student (gemma3 on the RPi) must keep prompt + reply within its 1024-token
# sliding window, so facts are rendered one line per room with short labels.
# Every ambient / contact / energy reading is kept; only device-internal chip
# temperatures ("... device" + temperature) are dropped (not room comfort).
# ---------------------------------------------------------------------------

STUDENT_ROOM_ORDER = ("bathroom", "hallway", "large_room", "small_room", "outside", "localization")

# Metrics guests ask about most; their absence is written out explicitly.
CORE_METRICS = ("temperature", "humidity", "pressure")

AMBIENT_ORDER = ("temperature", "humidity", "pressure", "voc", "occupancy", "illuminance")
AMBIENT_LABELS: Dict[str, str] = {
    "temperature": "temp.",
    "humidity": "wilg.",
    "pressure": "ciśn.",
    "voc": "VOC",
    "occupancy": "obecność",
    "illuminance": "oświetlenie",
}
# Short source tags, used only when a room has the same metric from >1 sensor.
SOURCE_TAGS: Dict[str, str] = {
    "climate sensor": "klimat",
    "air quality sensor": "jakość pow.",
    "motion sensor": "ruch",
}
ENERGY_NAMES: Dict[str, str] = {
    "boiler": "bojler",
    "heater": "grzejnik",
    "outlet": "gniazdko",
    "lights": "światła",
}
CONTACT_NAMES: Dict[str, Tuple[str, str]] = {
    # name -> (label, grammatical ending for zamknięt-/otwart-)
    "window": ("okno", "e"),
    "door": ("drzwi", "e"),
    "fridge": ("lodówka", "a"),
}


def _split_device(device_desc: str) -> Tuple[str, Optional[str], str, bool]:
    """'window 1 contact sensor device' -> ('window', '1', 'contact sensor', True)."""
    is_internal = device_desc.endswith(" device")
    base = device_desc[: -len(" device")] if is_internal else device_desc
    words = base.split()
    name = words[0] if words else base
    number = words[1] if len(words) > 1 and words[1].isdigit() else None
    kind = " ".join(words[2:] if number else words[1:])
    return name, number, kind, is_internal


def is_student_entry(device_desc: str, metric: str) -> bool:
    """Device-internal chip temperatures are not shown to the Student."""
    return not (_split_device(device_desc)[3] and metric == "temperature")


def energy_kwh(formatted_wh: str) -> str:
    """Cumulative meter '84458.8 Wh' -> '84.5 kWh' (digits are tokens; 0.1 Wh adds nothing)."""
    try:
        return f"{float(formatted_wh.split()[0]) / 1000:.1f} kWh"
    except (ValueError, IndexError):
        return formatted_wh


def _contact_text(label: str, ending: str, formatted: str) -> str:
    if formatted == "zamknięty":
        return f"{label} zamknięt{ending}"
    if formatted == "otwarty":
        return f"{label} otwart{ending}"
    return f"{label} {formatted}"


def _render_room(entries: List[Tuple[str, str, str, str]]) -> Tuple[str, List[Tuple[str, str, str, str]]]:
    """Render one room's entries; returns (text, entries actually rendered)."""
    rendered: List[Tuple[str, str, str, str]] = []
    parts: List[str] = []

    # Ambient metrics, merged across sensors.
    by_metric: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for e in entries:
        if e[2] in AMBIENT_LABELS:
            by_metric.setdefault(e[2], []).append(e)
    for metric in AMBIENT_ORDER:
        items = sorted(by_metric.get(metric, []), key=lambda e: e[1])
        if not items:
            continue
        values = []
        for e in items:
            name, _num, kind, _ = _split_device(e[1])
            tag = "szafka IoT" if name == "iot-cabinet" else (
                SOURCE_TAGS.get(kind, e[1]) if len(items) > 1 else None
            )
            values.append(f"{e[3]} ({tag})" if tag else e[3])
            rendered.append(e)
        sep = " " if metric != "occupancy" else ": "
        parts.append(f"{AMBIENT_LABELS[metric]}{sep}{', '.join(values)}")

    # Contacts (windows, doors, fridge) with optional fridge state.
    contacts = sorted((e for e in entries if e[2] in ("contact", "state")), key=lambda e: (e[1], e[2]))
    windows_in_room = {_split_device(e[1])[1] for e in contacts if _split_device(e[1])[0] == "window"}
    for e in contacts:
        name, num, _kind, _ = _split_device(e[1])
        label, ending = CONTACT_NAMES.get(name, (name, "y"))
        if name == "window" and len(windows_in_room) > 1 and num:
            label = f"{label} {num}"
        if e[2] == "contact":
            parts.append(_contact_text(label, ending, e[3]))
        else:
            parts.append(f"{label} stan {e[3]}")
        rendered.append(e)

    # Energy meters: "grzejnik: 234.5 W (licznik 84458.8 Wh)".
    meters: Dict[Tuple[str, Optional[str]], Dict[str, Tuple[str, str, str, str]]] = {}
    for e in entries:
        if e[2] in ("power", "energy"):
            name, num, _kind, _ = _split_device(e[1])
            meters.setdefault((name, num), {})[e[2]] = e
    for (name, num), m in sorted(meters.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        label = ENERGY_NAMES.get(name, name)
        if name == "outlet" and num:
            label = f"{label} {num}"
        power = m.get("power")
        energy = m.get("energy")
        if power and energy:
            parts.append(f"{label}: {power[3]} (licznik {energy_kwh(energy[3])})")
        elif power:
            parts.append(f"{label}: {power[3]}")
        elif energy:
            parts.append(f"{label}: licznik {energy_kwh(energy[3])}")
        rendered.extend(x for x in (power, energy) if x)

    return "; ".join(parts), rendered


def _short_timestamp(ts: str) -> str:
    """'2025-12-27 10:10:00+00:00' -> '2025-12-27 10:10 UTC'."""
    t = ts.strip()
    if t.endswith("+00:00"):
        t = t[: -len("+00:00")]
    if len(t) >= 19 and t[16:19] == ":00":
        t = t[:16]
    return f"{t} UTC"


def render_student_facts(
    snapshot: "SensorSnapshot",
    *,
    skip_rooms: Tuple[str, ...] = (),
) -> Tuple[str, List[Tuple[str, str, str, str]]]:
    """Compact Student facts + the list of entries rendered (for lossless tests)."""
    header = f"Stan mieszkania na {_short_timestamp(snapshot.timestamp)}."
    rooms = snapshot.apartment_rooms or []
    lines = [header]
    if rooms:
        lines.append(f"Pomieszczenia mieszkania: {', '.join(rooms)}.")
    student_entries = [e for e in snapshot.entries if is_student_entry(e[1], e[2])]

    rendered_all: List[Tuple[str, str, str, str]] = []
    room_keys = [k for k in STUDENT_ROOM_ORDER if any(e[0] == k for e in student_entries)]
    room_keys += sorted({e[0] for e in student_entries} - set(room_keys))
    # Apartment rooms are always listed, so a missing reading is stated, not implied
    # (A/B test: the Student invented numbers when a metric was simply absent).
    for key in APARTMENT_ROOM_KEYS:
        if key not in room_keys and ROOM_LABELS[key] in rooms:
            room_keys.insert(list(STUDENT_ROOM_ORDER).index(key), key)
    for key in room_keys:
        if key in skip_rooms:
            continue
        room_entries = [e for e in student_entries if e[0] == key]
        text, rendered = _render_room(room_entries)
        if key in APARTMENT_ROOM_KEYS:
            present = {e[2] for e in room_entries}
            missing = [AMBIENT_LABELS[m] for m in CORE_METRICS if m not in present]
            if not room_entries:
                text = "brak odczytów"
            elif missing:
                gap = "brak " + "/".join(missing)
                text = f"{text}; {gap}" if text else gap
        if text:
            lines.append(f"{ROOM_LABELS.get(key, key)}: {text}")
            rendered_all.extend(rendered)
    return "\n".join(lines), rendered_all


class SensorSimulator:
    """Simulates real-time sensor data by advancing through a CSV file.

    Each call to `next_snapshot()` moves the cursor forward by `step` rows
    (default 1 = 5 minutes of simulated time per refresh cycle)
    and returns exactly one coherent timestamped row.
    When the cursor reaches the end, it wraps around to the beginning.
    """

    def __init__(self, csv_path: Path, step: int = 1) -> None:
        self._csv_path = csv_path
        self._step = step

        self._header: List[str] = []
        self._rows: List[List[str]] = []
        self._col_info: List[Optional[tuple[str, str, str]]] = []
        self._apartment_rooms: List[str] = []
        self._cursor: int = 0
        self._loaded = False

    @property
    def apartment_rooms(self) -> List[str]:
        """Stable Polish labels of indoor rooms defined in the CSV schema."""
        self._load()
        return list(self._apartment_rooms)

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
        rooms_in_schema = {
            info[0] for info in self._col_info if info is not None and info[0] in APARTMENT_ROOM_KEYS
        }
        self._apartment_rooms = [
            ROOM_LABELS[key] for key in APARTMENT_ROOM_KEYS if key in rooms_in_schema
        ]
        self._cursor = 1 if self._rows else 0
        self._loaded = True
        logger.info(
            "CSV loaded: %d rows, rooms=%s, cursor starts at %d",
            len(self._rows),
            ",".join(self._apartment_rooms),
            self._cursor,
        )

    def next_snapshot(self) -> SensorSnapshot:
        """Advance cursor and return snapshot at current position."""
        self._load()

        if not self._rows:
            return SensorSnapshot(timestamp="?", readings={}, row_count=0, cursor=0, total_rows=0)

        row_number = min(max(self._cursor, 1), len(self._rows))
        snapshot = _build_snapshot(
            self._rows[row_number - 1],
            self._col_info,
            row_number=row_number,
            total_rows=len(self._rows),
        )
        snapshot.apartment_rooms = list(self._apartment_rooms)

        # Advance cursor
        self._cursor += self._step
        if self._cursor > len(self._rows):
            self._cursor = 1
            logger.info("CSV simulation wrapped around to the beginning.")

        return snapshot
