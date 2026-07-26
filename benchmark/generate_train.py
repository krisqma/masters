#!/usr/bin/env python3
"""
Generate data/train.jsonl for MLX chat fine-tuning from real db_sampler rows.

The script has two stages:
1. choose 400 auditable real cases into data/train_chosen.csv;
2. generate user/assistant pairs with Haiku via OpenRouter, while the system
   message is built deterministically from the selected CSV row.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR.parent / "db_sampler" / "sensor_data_2025-10-31_2026-01-27.csv"
DATA_DIR = BASE_DIR / "data"
CHOSEN_FILE = DATA_DIR / "train_chosen.csv"
OUTPUT_FILE = DATA_DIR / "train.jsonl"
RANDOM_SEED = 42
DEFAULT_HAIKU_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_GENERATOR_MODEL = "anthropic/claude-sonnet-4-6"
ACTIVE_POWER_W = 1.0
STALE_HOUR_CHOICES = [3, 6, 12, 24, 36, 48, 72, 96, 168]

CATEGORY_COUNTS = {
    "factual_status": 160,
    "missing_data": 80,
    "conflicting_data": 60,
    "stale_data": 40,
    "paraphrase": 60,
}

CHOSEN_FIELDS = [
    "case_id",
    "category",
    "source_row_index",
    "source_time",
    "target_sensor",
    "target_room",
    "target_metric",
    "target_value",
    "paraphrase_of",
    "omit_sensor",
    "stale_hours",
]

# Key sensor columns -> (room, metric, unit)
SENSOR_MAP = {
    "56_large_room_room_1_air_quality_sensor_temperature": ("salon", "temperatura (czujnik powietrza)", "°C"),
    "56_large_room_room_1_air_quality_sensor_humidity": ("salon", "wilgotność (czujnik powietrza)", "%"),
    "56_large_room_room_1_climate_sensor_temperature": ("salon", "temperatura (klimat)", "°C"),
    "56_large_room_room_1_climate_sensor_humidity": ("salon", "wilgotność (klimat)", "%"),
    "56_small_room_room_1_climate_sensor_temperature": ("sypialnia", "temperatura", "°C"),
    "56_small_room_room_1_climate_sensor_humidity": ("sypialnia", "wilgotność", "%"),
    "56_bathroom_room_1_climate_sensor_temperature": ("łazienka", "temperatura", "°C"),
    "56_bathroom_room_1_climate_sensor_humidity": ("łazienka", "wilgotność", "%"),
    "56_hallway_iot-cabinet_1_climate_sensor_temperature": ("przedpokój", "temperatura", "°C"),
    "56_hallway_iot-cabinet_1_climate_sensor_humidity": ("przedpokój", "wilgotność", "%"),
    "56_large_room_room_1_motion_sensor_occupancy": ("salon", "ruch/obecność", "motion"),
    "56_hallway_room_1_motion_sensor_occupancy": ("przedpokój", "ruch/obecność", "motion"),
    "56_bathroom_room_1_motion_sensor_occupancy": ("łazienka", "ruch/obecność", "motion"),
    "56_small_room_room_1_motion_sensor_occupancy": ("sypialnia", "ruch/obecność", "motion"),
    "56_large_room_window_1_contact_sensor_contact": ("salon", "okno 1", "contact"),
    "56_large_room_window_2_contact_sensor_contact": ("salon", "okno 2", "contact"),
    "56_bathroom_window_1_contact_sensor_contact": ("łazienka", "okno", "contact"),
    "56_small_room_window_1_contact_sensor_contact": ("sypialnia", "okno", "contact"),
    "56_outside_door_1_contact_sensor_contact": ("wejście", "drzwi", "contact"),
    "56_bathroom_boiler_1_energy_meter_power": ("łazienka", "bojler moc", "W"),
    "56_bathroom_heater_1_energy_meter_power": ("łazienka", "grzejnik moc", "W"),
    "56_small_room_heater_1_energy_meter_power": ("sypialnia", "grzejnik moc", "W"),
    "56_large_room_outlet_3_energy_meter_power": ("salon", "gniazdko 3 moc", "W"),
    "56_large_room_outlet_5_energy_meter_power": ("salon", "gniazdko 5 moc", "W"),
    "56_localization_lights_1_energy_meter_power": ("przedpokój", "oświetlenie moc", "W"),
}

SENSOR_COLS = list(SENSOR_MAP.keys())
CONFLICT_AQ_COL = "56_large_room_room_1_air_quality_sensor_temperature"
CONFLICT_CLIMATE_COL = "56_large_room_room_1_climate_sensor_temperature"


def _fmt_value(val: Any, unit: str) -> str:
    if unit == "motion":
        return "wykryto ruch" if float(val) > 0 else "brak ruchu"
    if unit == "contact":
        return "otwarte" if bool(val) else "zamknięte"
    return f"{float(val):.1f} {unit}"


def _fmt_target_value(row: pd.Series, sensor: str) -> str:
    _room, _metric, unit = SENSOR_MAP[sensor]
    return _fmt_value(row[sensor], unit)


def _timestamp(row: pd.Series) -> str:
    return row["_time"].strftime("%Y-%m-%d %H:%M:%S")


def load_and_filter(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Nie znaleziono CSV: {csv_path}")
    available = [c for c in SENSOR_COLS if c in pd.read_csv(csv_path, nrows=0).columns]
    df = pd.read_csv(csv_path, usecols=["_time"] + available, parse_dates=["_time"])
    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    df["_sensor_count"] = df[available].notna().sum(axis=1)
    return df


def build_system_prompt(
    row: pd.Series,
    omitted_sensor: str | None = None,
    stale_hours: float | None = None,
) -> str:
    ts = _timestamp(row)
    lines = [
        "Jesteś Wilga, zwięzły polski asystent smart-home dla mieszkania użytkownika.",
        "Odpowiadasz wyłącznie na podstawie danych z poniższego kontekstu.",
        "Informujesz o stanie mieszkania; nie wykonujesz akcji ani nie sterujesz urządzeniami.",
        "Odpowiadaj krótko, naturalnie i konkretnie: zwykle jednym zdaniem.",
        "Nie zgaduj, nie uśredniaj i nie dopowiadaj danych, których nie ma w kontekście.",
        "Jeśli brakuje danych, powiedz wprost, że nie masz tej informacji w aktualnym kontekście.",
        "Jeśli dane są sprzeczne, wskaż sprzeczność zamiast wybierać jedną wartość.",
        "Jeśli kontekst oznacza dane jako stare, zaznacz brak pewności co do aktualnego stanu.",
    ]
    if stale_hours is not None:
        lines.append(f"UWAGA: poniższe dane pochodzą sprzed {stale_hours:.0f} godzin. Mogą być nieaktualne.")
    lines.append(f"Stan mieszkania na {ts} UTC:")

    for col, (room, metric, unit) in SENSOR_MAP.items():
        if col == omitted_sensor or col not in row.index:
            continue
        val = row[col]
        if pd.notna(val):
            lines.append(f"- {room}: {metric} = {_fmt_value(val, unit)}")
    return "\n".join(lines)


def _available_sensors(row: pd.Series) -> list[str]:
    return [col for col in SENSOR_COLS if col in row.index and pd.notna(row[col])]


def _sample_indices(pool: pd.DataFrame, count: int, rng: random.Random, label: str) -> list[int]:
    if len(pool) < count:
        raise RuntimeError(f"Za mało kandydatów dla {label}: potrzeba {count}, jest {len(pool)}")
    return rng.sample(list(pool.index), count)


def _sensor_candidates(
    pool: pd.DataFrame,
    sensor: str,
    rng: random.Random,
    prefer_active_power: bool = True,
) -> list[int]:
    present = pool[pool[sensor].notna()]
    _room, _metric, unit = SENSOR_MAP[sensor]
    if unit == "W" and prefer_active_power:
        active = list(present[present[sensor].astype(float) > ACTIVE_POWER_W].index)
        inactive = list(present[present[sensor].astype(float) <= ACTIVE_POWER_W].index)
        rng.shuffle(active)
        rng.shuffle(inactive)
        return active + inactive
    indices = list(present.index)
    rng.shuffle(indices)
    return indices


def _balanced_present_cases(
    df: pd.DataFrame,
    pool: pd.DataFrame,
    count: int,
    category: str,
    case_prefix: str,
    rng: random.Random,
    stale: bool = False,
) -> list[dict[str, str]]:
    candidates = {
        sensor: _sensor_candidates(pool, sensor, rng)
        for sensor in SENSOR_COLS
        if sensor in pool.columns and pool[sensor].notna().any()
    }
    sensors = list(candidates)
    rng.shuffle(sensors)
    used_pairs: set[tuple[int, str]] = set()
    used_rows: set[int] = set()
    cases: list[dict[str, str]] = []
    positions = {sensor: 0 for sensor in sensors}

    while len(cases) < count:
        added = False
        for sensor in sensors:
            indices = candidates[sensor]
            idx: int | None = None
            while positions[sensor] < len(indices):
                candidate_idx = int(indices[positions[sensor]])
                positions[sensor] += 1
                if (candidate_idx, sensor) not in used_pairs and candidate_idx not in used_rows:
                    idx = candidate_idx
                    break
            if idx is None:
                for candidate_idx in indices:
                    candidate_idx = int(candidate_idx)
                    if (candidate_idx, sensor) not in used_pairs:
                        idx = candidate_idx
                        break
            if idx is None:
                continue

            row = df.loc[idx]
            room, metric, _unit = SENSOR_MAP[sensor]
            case_no = len(cases) + 1
            stale_hours = ""
            if stale:
                stale_hours = str(STALE_HOUR_CHOICES[(case_no - 1) % len(STALE_HOUR_CHOICES)])
            cases.append({
                "case_id": f"{case_prefix}_{case_no:03d}",
                "category": category,
                "source_row_index": str(idx),
                "source_time": _timestamp(row),
                "target_sensor": sensor,
                "target_room": room,
                "target_metric": metric,
                "target_value": _fmt_target_value(row, sensor),
                "paraphrase_of": "",
                "omit_sensor": "",
                "stale_hours": stale_hours,
            })
            used_pairs.add((idx, sensor))
            used_rows.add(idx)
            added = True
            if len(cases) >= count:
                break
        if not added:
            break

    if len(cases) != count:
        raise RuntimeError(f"Za mało przypadków dla {category}: {len(cases)}")
    return cases


def choose_cases(df: pd.DataFrame) -> list[dict[str, str]]:
    rng = random.Random(RANDOM_SEED)
    cases: list[dict[str, str]] = []

    good = df[df["_sensor_count"] >= 8]
    missing_base = df[df["_sensor_count"] >= 5]
    stale_pool = df[(df["_sensor_count"] >= 5) & (df["_time"] < pd.Timestamp("2025-12-01", tz="UTC"))]
    conflict_pool = df[
        df[CONFLICT_AQ_COL].notna()
        & df[CONFLICT_CLIMATE_COL].notna()
        & ((df[CONFLICT_AQ_COL] - df[CONFLICT_CLIMATE_COL]).abs() > 1.0)
    ]

    factual_cases = _balanced_present_cases(
        df,
        good,
        CATEGORY_COUNTS["factual_status"],
        "factual_status",
        "factual",
        rng,
    )
    cases.extend(factual_cases)

    missing_cases: list[dict[str, str]] = []
    missing_candidates: dict[str, list[int]] = {}
    missing_sensors = SENSOR_COLS[:]
    rng.shuffle(missing_sensors)
    for sensor in missing_sensors:
        indices = list(missing_base[missing_base[sensor].isna()].index)
        rng.shuffle(indices)
        if indices:
            missing_candidates[sensor] = indices

    round_no = 0
    while len(missing_cases) < CATEGORY_COUNTS["missing_data"]:
        added = False
        for sensor in missing_sensors:
            indices = missing_candidates.get(sensor, [])
            if round_no >= len(indices):
                continue
            idx = indices[round_no]
            row = df.loc[idx]
            room, metric, _unit = SENSOR_MAP[sensor]
            missing_cases.append({
                "case_id": f"missing_{len(missing_cases) + 1:03d}",
                "category": "missing_data",
                "source_row_index": str(idx),
                "source_time": _timestamp(row),
                "target_sensor": sensor,
                "target_room": room,
                "target_metric": metric,
                "target_value": "",
                "paraphrase_of": "",
                "omit_sensor": sensor,
                "stale_hours": "",
            })
            added = True
            if len(missing_cases) >= CATEGORY_COUNTS["missing_data"]:
                break
        if not added:
            break
        round_no += 1
    if len(missing_cases) != CATEGORY_COUNTS["missing_data"]:
        raise RuntimeError(f"Za mało przypadków missing_data: {len(missing_cases)}")
    rng.shuffle(missing_cases)
    cases.extend(missing_cases)

    conflict_indices = _sample_indices(conflict_pool, CATEGORY_COUNTS["conflicting_data"], rng, "conflicting_data")
    for i, idx in enumerate(conflict_indices, start=1):
        row = df.loc[idx]
        aq_val = _fmt_target_value(row, CONFLICT_AQ_COL)
        climate_val = _fmt_target_value(row, CONFLICT_CLIMATE_COL)
        cases.append({
            "case_id": f"conflict_{i:03d}",
            "category": "conflicting_data",
            "source_row_index": str(idx),
            "source_time": _timestamp(row),
            "target_sensor": f"{CONFLICT_AQ_COL}|{CONFLICT_CLIMATE_COL}",
            "target_room": "salon",
            "target_metric": "temperatura",
            "target_value": f"czujnik powietrza: {aq_val}; klimat: {climate_val}",
            "paraphrase_of": "",
            "omit_sensor": "",
            "stale_hours": "",
        })

    cases.extend(_balanced_present_cases(
        df,
        stale_pool,
        CATEGORY_COUNTS["stale_data"],
        "stale_data",
        "stale",
        rng,
        stale=True,
    ))

    for i, source_case in enumerate(factual_cases[:CATEGORY_COUNTS["paraphrase"]], start=1):
        paraphrase = dict(source_case)
        paraphrase["case_id"] = f"paraphrase_{i:03d}"
        paraphrase["category"] = "paraphrase"
        paraphrase["paraphrase_of"] = source_case["case_id"]
        cases.append(paraphrase)

    return cases


def write_chosen_csv(cases: list[dict[str, str]], path: Path = CHOSEN_FILE) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHOSEN_FIELDS)
        writer.writeheader()
        writer.writerows(cases)


def read_chosen_csv(path: Path = CHOSEN_FILE) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_chosen(cases: list[dict[str, str]], df: pd.DataFrame) -> None:
    counts = Counter(case["category"] for case in cases)
    if counts != Counter(CATEGORY_COUNTS):
        raise RuntimeError(f"Niepoprawny rozkład kategorii: {dict(counts)}")
    if len(cases) != sum(CATEGORY_COUNTS.values()):
        raise RuntimeError(f"Niepoprawna liczba przypadków: {len(cases)}")

    case_ids = [case["case_id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("case_id nie są unikalne")

    for case in cases:
        row = df.loc[int(case["source_row_index"])]
        category = case["category"]
        if category in {"factual_status", "stale_data", "paraphrase"}:
            sensor = case["target_sensor"]
            if sensor not in SENSOR_MAP or pd.isna(row[sensor]):
                raise RuntimeError(f"{case['case_id']}: target sensor nie ma wartości")
        if category == "missing_data":
            sensor = case["target_sensor"]
            if sensor not in SENSOR_MAP or pd.notna(row[sensor]):
                raise RuntimeError(f"{case['case_id']}: target missing_data nie jest pusty")
        if category == "conflicting_data":
            diff = abs(float(row[CONFLICT_AQ_COL]) - float(row[CONFLICT_CLIMATE_COL]))
            if diff <= 1.0:
                raise RuntimeError(f"{case['case_id']}: konflikt temperatury jest za mały ({diff:.2f})")
        if category == "stale_data" and not case["stale_hours"]:
            raise RuntimeError(f"{case['case_id']}: brak stale_hours")


def _strip_json_markdown(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    return text


GENERATOR_SYSTEM_PROMPT = """\
Jesteś ekspertem od tworzenia danych treningowych dla polskiego asystenta smart-home Wilga.
Tworzysz wyłącznie pytanie użytkownika i idealną krótką odpowiedź asystenta.
Nie wolno Ci zmieniać kontekstu systemowego ani wartości sensorów.
Priorytetem jest jakość danych fine-tuningowych: poprawna polszczyzna, brak dwuznaczności i pełna zgodność z kontekstem.
Odpowiadaj WYŁĄCZNIE surowym JSON object, bez markdown i bez komentarzy.\
"""

GENERATOR_USER_TEMPLATE = """\
Wygeneruj jeden przykład treningowy dla Wilgi.

KATEGORIA:
{category}

KONTEKST SYSTEMOWY, KTÓRY MODEL DOSTANIE W TRAKCIE FINE-TUNINGU:
{system_prompt}

CEL PYTANIA:
- pomieszczenie: {target_room}
- metryka: {target_metric}
- wartość / sytuacja: {target_value}
- doprecyzowanie celu: {target_hint}

REGUŁA KATEGORII:
{category_rule}

Wymagania:
- user: jedno naturalne pytanie po polsku, bez cytowania nazw technicznych kolumn.
- user musi respektować doprecyzowanie celu; jeśli cel mówi o czujniku powietrza, klimatu albo pierwszym/drugim oknie, pytanie musi zawierać to rozróżnienie.
- assistant: krótka odpowiedź Wilgi po polsku, zwykle jedno zdanie.
- assistant musi być w 100% zgodny z kontekstem systemowym.
- assistant nie może odpowiadać ogólnie "tej danej"; musi nazwać brakującą metrykę i pomieszczenie, jeśli kategoria to missing_data.
- dla ruchu/obecności pisz naturalnie: "wykryto ruch" albo "nie wykryto ruchu"; nie pisz "jest brak ruchu".
- dla konfliktów pisz "dwa rozbieżne odczyty" albo "rozbieżne odczyty"; nie pisz "dwie rozbieżne odczyty".
- dla stale_data podaj liczbę godzin dokładnie jako "{stale_hours_text}".
- jeśli metryka zawiera słowo "moc", pytaj o moc albo pobór mocy; nie używaj słów energia, energii ani zużycie.
- jeśli odpowiedź podaje wartość liczbową sensora, musi zawierać dokładnie ten ciąg znaków: {target_value}.
- nie dodawaj porad, wyjaśnień ani danych spoza kontekstu.
{paraphrase_instruction}

Zwróć dokładnie:
{{"user": "...", "assistant": "..."}}\
"""

CATEGORY_RULES = {
    "factual_status": (
        "Zapytaj o konkretną wartość obecną w kontekście. "
        "Odpowiedź ma podać tę wartość wprost."
    ),
    "missing_data": (
        "Zapytaj o wskazaną metrykę, której nie ma w kontekście. "
        "Odpowiedź ma jasno powiedzieć, że Wilga nie ma tej informacji w aktualnym kontekście."
    ),
    "conflicting_data": (
        "Zapytaj o temperaturę w salonie. Kontekst zawiera dwa rozbieżne odczyty. "
        "Odpowiedź ma wskazać sprzeczność i nie może wybierać jednej wartości."
    ),
    "stale_data": (
        "Zapytaj o wskazaną wartość. Kontekst oznacza dane jako stare. "
        "Odpowiedź ma zaznaczyć brak pewności co do aktualnego stanu."
    ),
    "paraphrase": (
        "To parafraza wcześniejszego przypadku. Zadaj pytanie inaczej niż typowe brzmienie, "
        "ale zachowaj ten sam sens i tę samą docelową odpowiedź."
    ),
}


def _target_hint(case: dict[str, str]) -> str:
    sensor = case["target_sensor"]
    metric = case["target_metric"]
    room = case["target_room"]
    if sensor == "56_large_room_room_1_air_quality_sensor_temperature":
        return "pytanie musi wskazywać temperaturę z czujnika powietrza w salonie"
    if sensor == "56_large_room_room_1_climate_sensor_temperature":
        return "pytanie musi wskazywać temperaturę z czujnika klimatu w salonie"
    if sensor == "56_large_room_room_1_air_quality_sensor_humidity":
        return "pytanie musi wskazywać wilgotność z czujnika powietrza w salonie"
    if sensor == "56_large_room_room_1_climate_sensor_humidity":
        return "pytanie musi wskazywać wilgotność z czujnika klimatu w salonie"
    if sensor == "56_large_room_window_1_contact_sensor_contact":
        return "pytanie musi wskazywać pierwsze okno w salonie"
    if sensor == "56_large_room_window_2_contact_sensor_contact":
        return "pytanie musi wskazywać drugie okno w salonie"
    if metric == "ruch/obecność":
        return f"pytanie dotyczy tego, czy w pomieszczeniu {room} wykryto ruch"
    if "moc" in metric:
        return f"pytanie dotyczy mocy/poboru mocy: {metric} w pomieszczeniu {room}"
    return f"pytanie dotyczy: {metric} w pomieszczeniu {room}"


def _contains_any(text: str, needles: list[str]) -> bool:
    text_l = text.lower()
    return any(needle in text_l for needle in needles)


def _missing_answer_names_target(case: dict[str, str], assistant: str) -> bool:
    assistant_l = assistant.lower()
    room_stems = {
        "salon": ["salon"],
        "łazienka": ["łazien", "lazien"],
        "sypialnia": ["sypial"],
        "przedpokój": ["przedpok"],
        "wejście": ["wej", "drzwi"],
    }
    metric_stems = {
        "temperatura": ["temperatur"],
        "temperatura (czujnik powietrza)": ["temperatur", "powietrz"],
        "temperatura (klimat)": ["temperatur", "klimat"],
        "wilgotność": ["wilgot"],
        "wilgotność (czujnik powietrza)": ["wilgot", "powietrz"],
        "wilgotność (klimat)": ["wilgot", "klimat"],
        "ruch/obecność": ["ruch", "obecn", "ktoś", "ktos"],
        "okno": ["okn"],
        "okno 1": ["okn", "pierwsz", "1"],
        "okno 2": ["okn", "drug", "2"],
        "drzwi": ["drzwi"],
        "bojler moc": ["bojler", "moc", "mocy"],
        "grzejnik moc": ["grzejnik", "moc", "mocy"],
        "gniazdko 3 moc": ["gniazdk", "trzec", "3", "moc", "mocy"],
        "gniazdko 5 moc": ["gniazdk", "piąt", "piat", "5", "moc", "mocy"],
        "oświetlenie moc": ["oświetl", "oswietl", "moc", "mocy"],
    }
    room_ok = any(stem in assistant_l for stem in room_stems.get(case["target_room"], [case["target_room"].lower()]))
    metric_ok = any(stem in assistant_l for stem in metric_stems.get(case["target_metric"], [case["target_metric"].split()[0].lower()]))
    return room_ok and metric_ok


def _disambiguate_user_question(case: dict[str, str], user: str) -> str:
    sensor = case["target_sensor"]
    user_l = user.lower()
    if sensor == "56_large_room_room_1_air_quality_sensor_temperature" and not _contains_any(user, ["powietrza", "czujnik"]):
        return "Jaka jest temperatura w salonie według czujnika powietrza?"
    if sensor == "56_large_room_room_1_climate_sensor_temperature" and not _contains_any(user, ["klimat", "czujnik klimatu"]):
        return "Jaka jest temperatura w salonie według czujnika klimatu?"
    if sensor == "56_large_room_room_1_air_quality_sensor_humidity" and not _contains_any(user, ["powietrza", "czujnik"]):
        return "Jaka jest wilgotność w salonie według czujnika powietrza?"
    if sensor == "56_large_room_room_1_climate_sensor_humidity" and not _contains_any(user, ["klimat", "czujnik klimatu"]):
        return "Jaka jest wilgotność w salonie według czujnika klimatu?"
    if sensor == "56_large_room_window_1_contact_sensor_contact" and not _contains_any(user, ["pierwsze", "pierwszy", "1"]):
        return "Czy pierwsze okno w salonie jest otwarte?"
    if sensor == "56_large_room_window_2_contact_sensor_contact" and not _contains_any(user, ["drugie", "drugi", "2"]):
        return "Czy drugie okno w salonie jest otwarte?"
    return user


def _assistant_contains_target_value(case: dict[str, str], assistant: str) -> bool:
    target_value = case["target_value"]
    if not target_value:
        return True
    if target_value in assistant:
        return True
    if case["target_metric"] == "ruch/obecność":
        assistant_l = assistant.lower()
        if target_value == "brak ruchu":
            return (
                "brak ruchu" in assistant_l
                or "nie wykryto ruchu" in assistant_l
                or "nie wykrył ruchu" in assistant_l
                or "nie wykrywa ruchu" in assistant_l
            )
        if target_value == "wykryto ruch":
            return (
                "wykryto ruch" in assistant_l
                or "wykrył ruch" in assistant_l
                or "wykrywa ruch" in assistant_l
            )
    return False


def _validate_generated_qa(case: dict[str, str], user: str, assistant: str, original_qa: dict[str, str] | None) -> None:
    if not user or not assistant:
        raise ValueError("empty user or assistant")

    user_l = user.lower()
    assistant_l = assistant.lower()
    metric = case["target_metric"]
    room = case["target_room"].lower()
    target_value = case["target_value"]

    if "moc" in metric and _contains_any(user, ["energia", "energii", "energię", "zużycie"]):
        raise ValueError("power question uses energy wording")

    if _contains_any(assistant, ["**", "```"]):
        raise ValueError("assistant uses markdown")

    if _contains_any(assistant, ["jest brak ruchu", "był brak ruchu"]):
        raise ValueError("unnatural motion wording")

    if "dwie rozbieżne odczyty" in assistant_l:
        raise ValueError("bad conflict grammar")

    sensor = case["target_sensor"]
    if sensor in {
        "56_large_room_room_1_air_quality_sensor_temperature",
        "56_large_room_room_1_air_quality_sensor_humidity",
    } and not _contains_any(user, ["czujnik powietrza", "powietrza"]):
        raise ValueError("ambiguous salon air-quality sensor question")

    if sensor in {
        "56_large_room_room_1_climate_sensor_temperature",
        "56_large_room_room_1_climate_sensor_humidity",
    } and not _contains_any(user, ["klimat", "czujnik klimatu"]):
        raise ValueError("ambiguous salon climate sensor question")

    if sensor == "56_large_room_window_1_contact_sensor_contact" and not _contains_any(user, ["pierwsze", "pierwszy", "1"]):
        raise ValueError("ambiguous first salon window question")

    if sensor == "56_large_room_window_2_contact_sensor_contact" and not _contains_any(user, ["drugie", "drugi", "2"]):
        raise ValueError("ambiguous second salon window question")

    if case["category"] == "missing_data":
        missing_markers = ["nie mam danych", "nie mam informacji", "brak danych", "brak informacji"]
        if not _contains_any(assistant, missing_markers):
            raise ValueError("missing_data answer lacks missing-data marker")
        if not _missing_answer_names_target(case, assistant):
            raise ValueError("missing_data answer does not name missing target")

    if case["category"] == "conflicting_data":
        if not _contains_any(assistant, ["sprzecz", "rozbież", "różne", "niejednoznacz"]):
            raise ValueError("conflicting_data answer lacks conflict marker")
        combined = f"{user} {assistant}".lower()
        if "salon" not in combined or "temperatur" not in combined:
            raise ValueError("conflicting_data answer does not name salon temperature")

    if case["category"] == "stale_data":
        if case["stale_hours"] and case["stale_hours"] not in assistant:
            raise ValueError("stale_data answer lacks exact stale hours")
        if not _contains_any(assistant, ["nieaktual", "brak pewności", "nie mam pewności", "aktualny stan może być inny"]):
            raise ValueError("stale_data answer lacks stale uncertainty")

    if case["category"] == "factual_status":
        if not _assistant_contains_target_value(case, assistant):
            raise ValueError("assistant does not contain exact target value")

    if original_qa is not None and assistant != original_qa["assistant"]:
        raise ValueError("paraphrase assistant differs from original")


def _api_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def generate_user_assistant(
    client: OpenAI,
    model: str,
    case: dict[str, str],
    system_prompt: str,
    original_qa: dict[str, str] | None = None,
) -> dict[str, str]:
    paraphrase_instruction = ""
    if original_qa is not None:
        paraphrase_instruction = (
            f'- to jest parafraza pytania "{original_qa["user"]}"; '
            f'assistant musi być dokładnie taki tekst: "{original_qa["assistant"]}".'
        )

    prompt = GENERATOR_USER_TEMPLATE.format(
        category=case["category"],
        system_prompt=system_prompt,
        target_room=case["target_room"],
        target_metric=case["target_metric"],
        target_value=case["target_value"] or "brak tej danej w kontekście",
        target_hint=_target_hint(case),
        category_rule=CATEGORY_RULES[case["category"]],
        stale_hours_text=case["stale_hours"] or "nie dotyczy",
        paraphrase_instruction=paraphrase_instruction,
    )

    raw = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.25,
            )
            raw = resp.choices[0].message.content or ""
            parsed = json.loads(_strip_json_markdown(raw))
            user = str(parsed["user"]).strip()
            assistant = str(parsed["assistant"]).strip()
            user = _disambiguate_user_question(case, user)
            if original_qa is not None:
                assistant = original_qa["assistant"]
            if not user or not assistant:
                raise ValueError("empty user or assistant")
            return {"user": user, "assistant": assistant}
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"{case['case_id']}: generation failed: {exc}; raw={raw!r}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def build_record(case: dict[str, str], row: pd.Series, qa: dict[str, str]) -> dict[str, Any]:
    stale_hours = float(case["stale_hours"]) if case["stale_hours"] else None
    omitted_sensor = case["omit_sensor"] or None
    system_prompt = build_system_prompt(row, omitted_sensor=omitted_sensor, stale_hours=stale_hours)
    return {
        "category": case["category"],
        "case_id": case["case_id"],
        "paraphrase_of": case["paraphrase_of"] or None,
        "source_time": case["source_time"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qa["user"]},
            {"role": "assistant", "content": qa["assistant"]},
        ],
    }


def validate_jsonl(path: Path = OUTPUT_FILE) -> None:
    counts: Counter[str] = Counter()
    total = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            counts[item["category"]] += 1
            messages = item["messages"]
            roles = [m["role"] for m in messages]
            if roles != ["system", "user", "assistant"]:
                raise RuntimeError(f"Linia {line_no}: niepoprawne role {roles}")
            if not all(m["content"].strip() for m in messages):
                raise RuntimeError(f"Linia {line_no}: pusta treść w messages")
    if total != sum(CATEGORY_COUNTS.values()):
        raise RuntimeError(f"Niepoprawna liczba rekordów JSONL: {total}")
    if counts != Counter(CATEGORY_COUNTS):
        raise RuntimeError(f"Niepoprawny rozkład JSONL: {dict(counts)}")


def generate_jsonl(
    cases: list[dict[str, str]],
    df: pd.DataFrame,
    output_file: Path,
    limit: int | None = None,
    force: bool = False,
    resume: bool = False,
    model_override: str | None = None,
) -> None:
    if output_file.exists() and not force and not resume:
        raise RuntimeError(f"{output_file} już istnieje. Użyj --force, żeby nadpisać.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = (
        model_override
        or os.environ.get("TRAIN_GENERATOR_MODEL")
        or os.environ.get("SONNET_MODEL")
        or DEFAULT_GENERATOR_MODEL
    )
    client = _api_client()
    selected = cases[:limit] if limit is not None else cases
    total = len(selected)
    generated_by_case_id: dict[str, dict[str, str]] = {}
    completed_case_ids: set[str] = set()

    if resume and output_file.exists():
        with output_file.open(encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item["messages"]
                generated_by_case_id[item["case_id"]] = {
                    "user": messages[1]["content"],
                    "assistant": messages[2]["content"],
                }
                completed_case_ids.add(item["case_id"])
        print(f"Wznawiam: pomijam {len(completed_case_ids)} gotowych rekordów z {output_file}")

    mode = "a" if resume and output_file.exists() else "w"
    with output_file.open(mode, encoding="utf-8") as f:
        for i, case in enumerate(selected, start=1):
            if case["case_id"] in completed_case_ids:
                continue
            row = df.loc[int(case["source_row_index"])]
            stale_hours = float(case["stale_hours"]) if case["stale_hours"] else None
            omitted_sensor = case["omit_sensor"] or None
            system_prompt = build_system_prompt(row, omitted_sensor=omitted_sensor, stale_hours=stale_hours)
            original_qa = None
            if case["category"] == "paraphrase":
                original_qa = generated_by_case_id.get(case["paraphrase_of"])
                if original_qa is None:
                    raise RuntimeError(f"{case['case_id']}: brak wygenerowanego oryginału {case['paraphrase_of']}")
            qa = generate_user_assistant(client, model, case, system_prompt, original_qa=original_qa)
            record = build_record(case, row, qa)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            generated_by_case_id[case["case_id"]] = qa
            print(f"[{i}/{total}] category={case['category']}")


def print_counts(cases: list[dict[str, str]]) -> None:
    counts = Counter(case["category"] for case in cases)
    for category in CATEGORY_COUNTS:
        print(f"  {category}: {counts[category]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data/train.jsonl for Wilga fine-tuning")
    parser.add_argument("--choose-only", action="store_true", help="Only create and validate data/train_chosen.csv")
    parser.add_argument("--generate-only", action="store_true", help="Use existing data/train_chosen.csv")
    parser.add_argument("--limit", type=int, help="Generate only first N JSONL records for a smoke test")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--resume", action="store_true", help="Append missing records to an existing JSONL file")
    parser.add_argument("--model", help="Override generator model; defaults to TRAIN_GENERATOR_MODEL, SONNET_MODEL, or Sonnet")
    parser.add_argument("--chosen-file", type=Path, default=CHOSEN_FILE)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    df = load_and_filter()

    if args.generate_only:
        cases = read_chosen_csv(args.chosen_file)
    else:
        if args.chosen_file.exists() and not args.force:
            raise RuntimeError(f"{args.chosen_file} już istnieje. Użyj --force, żeby nadpisać.")
        cases = choose_cases(df)
        validate_chosen(cases, df)
        write_chosen_csv(cases, args.chosen_file)
        print(f"Zapisano {len(cases)} przypadków -> {args.chosen_file}")
        print("Rozkład kategorii:")
        print_counts(cases)

    validate_chosen(cases, df)
    if args.choose_only:
        return

    generate_jsonl(
        cases,
        df,
        args.output_file,
        limit=args.limit,
        force=args.force,
        resume=args.resume,
        model_override=args.model,
    )
    if args.limit is None:
        validate_jsonl(args.output_file)
        print(f"Zapisano i zwalidowano {sum(CATEGORY_COUNTS.values())} rekordów -> {args.output_file}")
    else:
        print(f"Zapisano smoke-test {args.limit} rekordów -> {args.output_file}")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        sys.exit(1)
