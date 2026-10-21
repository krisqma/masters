"""Compact Student facts: lossless vs the legacy format and within the token budget."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.context_engine import STUDENT_STATIC_PREFIX, build_dynamic_context  # noqa: E402
from app.data_source import (  # noqa: E402
    ROOM_LABELS,
    SensorSimulator,
    energy_kwh,
    is_student_entry,
    render_student_facts,
)
from app.teacher_client import MAX_FINDINGS_CHARS  # noqa: E402

CSV = ROOT.parent / "db_sampler" / "sensor_data_2025-10-31_2026-01-27.csv"

# Mirrors whisper-proj/backend/app/student_prompt.py (estimate + budget).
WINDOW, MARGIN, NUM_PREDICT, MSG_OVERHEAD = 1024, 32, 128, 6
MIN_HISTORY_AND_QUESTION = 120


def _estimate(text: str) -> int:
    # Same calibrated estimate as student_prompt.estimate_tokens.
    if not text:
        return 0
    digits = sum(ch.isdigit() for ch in text)
    dense = digits / len(text) > 0.06
    return digits + math.ceil((len(text) - digits) / (1.95 if dense else 2.8))


@pytest.fixture(scope="module")
def snapshots():
    sim = SensorSimulator(CSV)
    sim.next_snapshot()  # load
    total = len(sim._rows)
    out = []
    for row in range(1, total + 1):
        sim._cursor = row
        out.append(sim.next_snapshot())
    return out


def _legacy_triples(text: str) -> set:
    """Parse the legacy '- Room: device – label = value' lines."""
    triples = set()
    for line in text.splitlines():
        m = re.match(r"- (.+?): (.+) – (.+?) = (.+)$", line)
        if m:
            room, device, label, value = m.groups()
            triples.add((room, device, label, value))
    return triples


def test_every_student_reading_rendered_exactly_once(snapshots):
    for snap in snapshots:
        _text, rendered = render_student_facts(snap)
        expected = [e for e in snap.entries if is_student_entry(e[1], e[2])]
        assert sorted(rendered) == sorted(expected), snap.timestamp


def test_compact_values_match_legacy_format(snapshots):
    """Every legacy line (minus device chip temperatures) has its value in the compact room line."""
    for snap in snapshots[::7]:
        compact, _ = render_student_facts(snap)
        room_lines = {ln.split(":", 1)[0]: ln for ln in compact.splitlines() if ":" in ln}
        for room, device, _label, value in _legacy_triples(snap.to_student_facts_text_verbose()):
            if device.endswith(" device"):
                continue
            line = room_lines.get(room)
            assert line is not None, (snap.timestamp, room)
            shown = value
            if value in ("zamknięty", "otwarty"):
                shown = value[:-1]  # zamknięt-e/-a, otwart-e/-a
            if value.endswith(" Wh"):
                shown = energy_kwh(value)  # cumulative meters shown in kWh
            assert shown in line, (snap.timestamp, room, device, value)


def test_device_chip_temperatures_dropped(snapshots):
    for snap in snapshots[::50]:
        compact, _ = render_student_facts(snap)
        assert "device" not in compact


def test_apartment_rooms_and_header_kept(snapshots):
    snap = snapshots[100]
    compact, _ = render_student_facts(snap)
    assert compact.startswith("Stan mieszkania na ")
    assert "Pomieszczenia mieszkania: " + ", ".join(snap.apartment_rooms) in compact
    for key in ("bathroom", "large_room"):
        assert ROOM_LABELS[key] in compact


def test_history_room_left_for_typical_rows(snapshots):
    """Without findings, nearly every row leaves room for a question and history.

    Rows that do not fit are degraded by the backend (student_prompt.ensure_prefix_fits,
    covered by its own test over every CSV row).
    """
    budget = WINDOW - MARGIN - NUM_PREDICT
    static_tokens = _estimate(STUDENT_STATIC_PREFIX) + MSG_OVERHEAD
    left = sorted(
        budget - static_tokens - _estimate(build_dynamic_context(facts_text=s.to_student_facts_text(), findings=None)) - MSG_OVERHEAD
        for s in snapshots
    )
    share_ok = sum(x >= MIN_HISTORY_AND_QUESTION for x in left) / len(left)
    # Without any degradation; the backend drops counters/findings for the rest.
    assert share_ok >= 0.60, share_ok
    assert left[len(left) // 2] >= 120, left[len(left) // 2]


def test_static_prefix_size():
    # Measured on the RPi: ~360 tokens (was 596 before the fix).
    assert _estimate(STUDENT_STATIC_PREFIX) <= 400


def test_findings_cap():
    from app.teacher_client import cap_findings

    long = "- Łazienka: wilgotność 61.8 % wysoka.\n- Duży pokój: VOC 261 ppb.\n- Na zewnątrz drzwi otwarte od dłuższego czasu, a temperatura spada."
    capped = cap_findings(long)
    assert len(capped) <= MAX_FINDINGS_CHARS
    assert capped.startswith("- Łazienka")


def test_missing_core_readings_are_explicit(snapshots):
    """Row 12000 has no bathroom temperature: the facts must say so (A/B: absent = invented)."""
    snap = snapshots[12000 - 1]
    compact, _ = render_student_facts(snap)
    lines = {ln.split(":", 1)[0]: ln for ln in compact.splitlines()}
    assert "brak temp./wilg./ciśn." in lines["Łazienka"]
    assert lines["Korytarz"].endswith("brak odczytów")
