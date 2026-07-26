#!/usr/bin/env python3
"""
Generate golden.jsonl from real sensor data (db_sampler CSV) + Sonnet question/answer generation.
"""
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CSV_PATH = Path(__file__).parent.parent / "db_sampler" / "sensor_data_2025-10-31_2026-01-27.csv"
OUTPUT_FILE = Path(__file__).parent / "golden.jsonl"
SONNET_MODEL = "anthropic/claude-sonnet-4-6"
RANDOM_SEED = 42

REQUIRED_FIELDS = {"case_id", "category", "system_prompt", "question", "golden_answer", "paraphrase_of"}

# Key sensor columns → (pomieszczenie, metryka, jednostka)
SENSOR_MAP = {
    "56_large_room_room_1_air_quality_sensor_temperature": ("salon", "temperatura (czujnik powietrza)", "°C"),
    "56_large_room_room_1_air_quality_sensor_humidity":    ("salon", "wilgotność (czujnik powietrza)", "%"),
    "56_large_room_room_1_climate_sensor_temperature":     ("salon", "temperatura (klimat)", "°C"),
    "56_large_room_room_1_climate_sensor_humidity":        ("salon", "wilgotność (klimat)", "%"),
    "56_small_room_room_1_climate_sensor_temperature":     ("sypialnia", "temperatura", "°C"),
    "56_small_room_room_1_climate_sensor_humidity":        ("sypialnia", "wilgotność", "%"),
    "56_bathroom_room_1_climate_sensor_temperature":       ("łazienka", "temperatura", "°C"),
    "56_bathroom_room_1_climate_sensor_humidity":          ("łazienka", "wilgotność", "%"),
    "56_hallway_iot-cabinet_1_climate_sensor_temperature": ("przedpokój", "temperatura", "°C"),
    "56_hallway_iot-cabinet_1_climate_sensor_humidity":    ("przedpokój", "wilgotność", "%"),
    "56_large_room_room_1_motion_sensor_occupancy":        ("salon", "ruch/obecność", "motion"),
    "56_hallway_room_1_motion_sensor_occupancy":           ("przedpokój", "ruch/obecność", "motion"),
    "56_bathroom_room_1_motion_sensor_occupancy":          ("łazienka", "ruch/obecność", "motion"),
    "56_small_room_room_1_motion_sensor_occupancy":        ("sypialnia", "ruch/obecność", "motion"),
    "56_large_room_window_1_contact_sensor_contact":       ("salon", "okno 1", "contact"),
    "56_large_room_window_2_contact_sensor_contact":       ("salon", "okno 2", "contact"),
    "56_bathroom_window_1_contact_sensor_contact":         ("łazienka", "okno", "contact"),
    "56_small_room_window_1_contact_sensor_contact":       ("sypialnia", "okno", "contact"),
    "56_outside_door_1_contact_sensor_contact":            ("wejście", "drzwi", "contact"),
    "56_bathroom_boiler_1_energy_meter_power":             ("łazienka", "bojler moc", "W"),
    "56_bathroom_heater_1_energy_meter_power":             ("łazienka", "grzejnik moc", "W"),
    "56_small_room_heater_1_energy_meter_power":           ("sypialnia", "grzejnik moc", "W"),
    "56_large_room_outlet_3_energy_meter_power":           ("salon", "gniazdko 3 moc", "W"),
    "56_large_room_outlet_5_energy_meter_power":           ("salon", "gniazdko 5 moc", "W"),
    "56_localization_lights_1_energy_meter_power":         ("przedpokój", "oświetlenie moc", "W"),
}

SENSOR_COLS = list(SENSOR_MAP.keys())


def _fmt_value(val, unit: str) -> str:
    if unit == "motion":
        return "wykryto ruch" if float(val) > 0 else "brak ruchu"
    if unit == "contact":
        return "otwarte" if bool(val) else "zamknięte"
    return f"{float(val):.1f} {unit}"


def build_system_prompt(row: pd.Series, ts: str, stale_hours: float | None = None) -> str:
    if stale_hours is not None:
        header = (
            f"Jesteś asystentem smart-home. "
            f"UWAGA: poniższe dane pochodzą sprzed {stale_hours:.0f} godzin "
            f"(ostatni odczyt: {ts} UTC). Dane mogą być nieaktualne.\nStan mieszkania:"
        )
    else:
        header = f"Jesteś asystentem smart-home. Stan mieszkania na {ts} UTC:"

    lines = [header]
    for col, (room, metric, unit) in SENSOR_MAP.items():
        if col not in row.index:
            continue
        val = row[col]
        if pd.notna(val):
            lines.append(f"- {room}: {metric} = {_fmt_value(val, unit)}")
    return "\n".join(lines)


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    available = [c for c in SENSOR_COLS if c in pd.read_csv(csv_path, nrows=0).columns]
    df = pd.read_csv(csv_path, usecols=["_time"] + available, parse_dates=["_time"])
    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    df["_sensor_count"] = df[available].notna().sum(axis=1)
    return df


def sample_cases(df: pd.DataFrame) -> list[dict]:
    rng = random.Random(RANDOM_SEED)

    good = df[df["_sensor_count"] >= 8]
    sparse = df[(df["_sensor_count"] >= 3) & (df["_sensor_count"] < 8)]
    early = df[df["_time"] < pd.Timestamp("2025-12-01", tz="UTC")]

    aq_col = "56_large_room_room_1_air_quality_sensor_temperature"
    cl_col = "56_large_room_room_1_climate_sensor_temperature"
    if aq_col in df.columns and cl_col in df.columns:
        conflict = df[
            df[aq_col].notna() & df[cl_col].notna() & (abs(df[aq_col] - df[cl_col]) > 1.0)
        ]
    else:
        conflict = good

    def pick(pool: pd.DataFrame, n: int) -> list[pd.Series]:
        pool = pool if len(pool) >= n else good
        indices = rng.sample(list(pool.index), min(n, len(pool)))
        return [pool.loc[i] for i in indices]

    cases = []
    factual_rows = pick(good, 8)
    factual_ids = []

    # 8x factual_status
    for i, row in enumerate(factual_rows):
        ts = row["_time"].strftime("%Y-%m-%d %H:%M:%S")
        cid = f"factual_{i+1:02d}"
        factual_ids.append(cid)
        cases.append({
            "case_id": cid,
            "category": "factual_status",
            "system_prompt": build_system_prompt(row, ts),
            "paraphrase_of": None,
        })

    # 4x missing_data
    for i, row in enumerate(pick(sparse, 4)):
        ts = row["_time"].strftime("%Y-%m-%d %H:%M:%S")
        cases.append({
            "case_id": f"missing_{i+1:02d}",
            "category": "missing_data",
            "system_prompt": build_system_prompt(row, ts),
            "paraphrase_of": None,
        })

    # 3x conflicting_data
    for i, row in enumerate(pick(conflict, 3)):
        ts = row["_time"].strftime("%Y-%m-%d %H:%M:%S")
        cases.append({
            "case_id": f"conflict_{i+1:02d}",
            "category": "conflicting_data",
            "system_prompt": build_system_prompt(row, ts),
            "paraphrase_of": None,
        })

    # 2x stale_data — early rows, explicit staleness note
    ref_time = pd.Timestamp("2025-11-15 14:00:00", tz="UTC")
    for i, row in enumerate(pick(early, 2)):
        ts = row["_time"].strftime("%Y-%m-%d %H:%M:%S")
        stale_h = (ref_time - row["_time"]).total_seconds() / 3600
        stale_h = max(stale_h, 3.0)
        cases.append({
            "case_id": f"stale_{i+1:02d}",
            "category": "stale_data",
            "system_prompt": build_system_prompt(row, ts, stale_hours=stale_h),
            "paraphrase_of": None,
        })

    # 3x paraphrase — same system_prompts as first 3 factual cases
    for i in range(3):
        orig = cases[i]
        cases.append({
            "case_id": f"paraphrase_{i+1:02d}",
            "category": "paraphrase",
            "system_prompt": orig["system_prompt"],
            "paraphrase_of": factual_ids[i],
        })

    return cases


_SONNET_SYSTEM = """\
Jesteś ekspertem od ewaluacji systemów RAG dla asystentów smart-home.
Masz przygotowane przypadki testowe z prawdziwymi odczytami czujników.
Każdy przypadek zawiera: case_id, category, system_prompt (prawdziwe dane).
Twoim zadaniem: wygenerować question i golden_answer dla każdego przypadku.
Odpowiadaj WYŁĄCZNIE surowym JSON array — zero komentarzy, zero markdown.\
"""

_SONNET_USER_TEMPLATE = """\
Poniżej {n} przypadków testowych. Wygeneruj question i golden_answer dla każdego.

Wymagania per kategoria:
- factual_status: pytanie o konkretną wartość OBECNĄ w system_prompt (1 zdanie); odpowiedź podaje tę wartość wprost (1-2 zdania)
- missing_data: pytanie o sensor którego NIE MA w system_prompt; golden_answer: "Nie mam danych o [X] w aktualnym kontekście."
- conflicting_data: system_prompt zawiera DWA czujniki temperatury salonu z różnymi wartościami — pytaj o temperaturę salonu; golden_answer: "Kontekst zawiera sprzeczne dane dotyczące temperatury w salonie."
- stale_data: pytanie o dowolną wartość; golden_answer: "Dane mogą być nieaktualne — ostatni odczyt pochodzi sprzed [N] godzin." (użyj N z system_prompt)
- paraphrase: INNE sformułowanie pytania niż oryginał (case_id oryginału w paraphrase_of); golden_answer identyczna jak odpowiedź dla oryginału

Zwróć JSON array z dokładnie {n} obiektami:
[{{"case_id": "...", "question": "...", "golden_answer": "..."}}]

Przypadki:
{cases_json}\
"""


def generate_qa(client: OpenAI, cases: list[dict]) -> dict[str, dict]:
    cases_json = json.dumps(
        [{"case_id": c["case_id"], "category": c["category"], "system_prompt": c["system_prompt"],
          "paraphrase_of": c["paraphrase_of"]} for c in cases],
        ensure_ascii=False, indent=2
    )
    user_prompt = _SONNET_USER_TEMPLATE.format(n=len(cases), cases_json=cases_json)

    resp = client.chat.completions.create(
        model=SONNET_MODEL,
        messages=[
            {"role": "system", "content": _SONNET_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )
    raw = resp.choices[0].message.content or ""

    # Strip markdown if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    parsed = json.loads(raw)
    return {item["case_id"]: item for item in parsed}


def run() -> None:
    if not CSV_PATH.exists():
        print(f"Błąd: nie znaleziono {CSV_PATH}")
        sys.exit(1)

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    print(f"Wczytuję dane z {CSV_PATH.name}...")
    df = load_and_filter(CSV_PATH)
    print(f"  {len(df)} wierszy, wiersze z ≥8 czujników: {(df['_sensor_count'] >= 8).sum()}")

    print("Próbkuję wiersze i buduję system_prompts...")
    cases = sample_cases(df)
    print(f"  {len(cases)} przypadków gotowych")

    print(f"Generuję question + golden_answer przez {SONNET_MODEL}...")
    qa_map = generate_qa(client, cases)

    records = []
    for case in cases:
        cid = case["case_id"]
        qa = qa_map.get(cid, {})
        record = {
            "case_id": cid,
            "category": case["category"],
            "system_prompt": case["system_prompt"],
            "question": qa.get("question", ""),
            "golden_answer": qa.get("golden_answer", ""),
            "paraphrase_of": case["paraphrase_of"],
        }
        missing = {f for f in REQUIRED_FIELDS if not record.get(f) and f != "paraphrase_of"}
        if missing:
            print(f"  WARN {cid}: brakuje pól {missing}")
        records.append(record)

    OUTPUT_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    )
    print(f"\nZapisano {len(records)} rekordów → {OUTPUT_FILE.name}")

    from collections import Counter
    counts = Counter(r["category"] for r in records)
    print("\nRozkład kategorii:")
    for cat in sorted(counts):
        print(f"  {cat}: {counts[cat]}")


if __name__ == "__main__":
    run()
