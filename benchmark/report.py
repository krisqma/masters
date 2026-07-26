#!/usr/bin/env python3
"""Generates report.html from results/*.jsonl files."""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles


def _load_records(results_dir: str) -> list[dict]:
    records = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _aggregate(records: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    overview = []
    latency_box = {}
    quality_bars = {}
    scatter = []
    top_params = {}

    for model, recs in by_model.items():
        scores = [r["composite_score"] for r in recs]
        latencies = [r["e2e_ms"] for r in recs]
        faithfulness = [r["faithfulness"] for r in recs]
        relevancy = [r["answer_relevancy"] for r in recs]
        conciseness = [r["conciseness"] for r in recs]
        polish = [r["polish_quality"] for r in recs]

        avg_composite = round(mean(scores), 4)
        avg_faith = round(mean(faithfulness), 4)
        avg_lat = round(mean(latencies), 1)

        ttft_vals = [
            float(r["ttft_ms"]) for r in recs if r.get("ttft_ms") is not None
        ]
        avg_ttft = round(mean(ttft_vals), 1) if ttft_vals else None

        # best_trial_score = max of per-trial averages (not single-case max)
        by_trial: dict[int, list[float]] = {}
        for r in recs:
            by_trial.setdefault(r["trial"], []).append(r["composite_score"])
        best_score = round(max(mean(v) for v in by_trial.values()), 4)

        overview.append({
            "model": model,
            "avg_composite": avg_composite,
            "avg_faithfulness": avg_faith,
            "avg_latency_ms": avg_lat,
            "avg_ttft_ms": avg_ttft,
            "best_trial_score": best_score,
        })

        sorted_lat = sorted(latencies)
        q = quantiles(sorted_lat, n=4) if len(sorted_lat) >= 4 else [sorted_lat[0]] * 3
        latency_box[model] = {
            "min": round(min(sorted_lat), 1),
            "p25": round(q[0], 1),
            "median": round(median(sorted_lat), 1),
            "p75": round(q[2], 1),
            "max": round(max(sorted_lat), 1),
        }

        quality_bars[model] = {
            "faithfulness": avg_faith,
            "answer_relevancy": round(mean(relevancy), 4),
            "conciseness": round(mean(conciseness), 4),
            "polish_quality": round(mean(polish), 4),
        }

        scatter.append({
            "model": model,
            "avg_e2e_ms": avg_lat,
            "avg_faithfulness": avg_faith,
        })

        # top 3 unique trials by composite score
        by_trial: dict[int, list[dict]] = defaultdict(list)
        for r in recs:
            by_trial[r["trial"]].append(r)
        trial_scores = [
            (trial_num, mean(t["composite_score"] for t in trecs), trecs[0])
            for trial_num, trecs in by_trial.items()
        ]
        trial_scores.sort(key=lambda x: x[1], reverse=True)
        top3 = []
        for _, score, sample in trial_scores[:3]:
            top3.append({
                "model": model,
                "trial": sample["trial"],
                "temperature": sample["temperature"],
                "num_ctx": sample["num_ctx"],
                "num_predict": sample["num_predict"],
                "composite_score": round(score, 4),
            })
        top_params[model] = top3

    overview.sort(key=lambda x: x["avg_composite"], reverse=True)
    return {
        "overview": overview,
        "latency_box": latency_box,
        "quality_bars": quality_bars,
        "scatter": scatter,
        "top_params": top_params,
        "records": records,
    }


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wilga Benchmark Report — {timestamp}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f5f5f5; color: #222; padding: 24px; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 8px; }}
  h2 {{ font-size: 1.1rem; margin: 32px 0 12px; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 32px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th {{ background: #f0f0f0; text-align: left; padding: 10px 12px; font-size: 0.85rem; color: #555; }}
  td {{ padding: 10px 12px; font-size: 0.9rem; border-top: 1px solid #eee; }}
  tr:hover td {{ background: #fafafa; }}
  .best {{ font-weight: 700; color: #1a6e3c; }}
  .chart-wrap {{ background: #fff; border-radius: 6px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 24px; }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  canvas {{ max-height: 320px; }}
  .filters {{ display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
  select {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.9rem; }}
  .tag-true {{ color: #c0392b; font-weight: 600; }}
  .tag-false {{ color: #27ae60; }}
  .trunc {{ max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: inline-block; }}
  .pagination {{ margin-top: 12px; display: flex; gap: 8px; align-items: center; }}
  .pagination button {{ padding: 4px 10px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; background: #fff; }}
  .pagination button:hover {{ background: #f0f0f0; }}
  #page-info {{ font-size: 0.85rem; color: #666; }}
</style>
</head>
<body>
<h1>Wilga Benchmark Report</h1>
<div class="subtitle">{timestamp} &nbsp;|&nbsp; {n_records} rekordów &nbsp;|&nbsp; {n_models} modeli</div>

<h2>1. Overview — ranking modeli</h2>
<table id="overview-table">
  <thead><tr>
    <th>Model</th>
    <th>Avg Composite</th>
    <th>Avg Faithfulness</th>
    <th>Avg E2E (ms)</th>
    <th>Avg TTFT (ms)</th>
    <th>Best Trial Score</th>
  </tr></thead>
  <tbody id="overview-body"></tbody>
</table>

<h2>2. Latency</h2>
<div class="chart-wrap"><canvas id="latencyChart"></canvas></div>

<h2>3. Quality metrics</h2>
<div class="chart-wrap"><canvas id="qualityChart"></canvas></div>

<h2>4. Trade-off: latency vs faithfulness</h2>
<div class="chart-wrap" style="max-width:600px"><canvas id="scatterChart"></canvas></div>

<h2>5. Top 3 konfiguracje per model</h2>
<table>
  <thead><tr>
    <th>Model</th><th>Trial</th><th>Temperature</th><th>num_ctx</th><th>num_predict</th><th>Composite Score</th>
  </tr></thead>
  <tbody id="params-body"></tbody>
</table>

<h2>6. Drilldown</h2>
<div class="filters">
  <select id="f-model"><option value="">Wszystkie modele</option></select>
  <select id="f-cat"><option value="">Wszystkie kategorie</option></select>
  <select id="f-hall"><option value="">Hallucination: wszystkie</option><option value="true">true</option><option value="false">false</option></select>
</div>
<table>
  <thead><tr>
    <th>case_id</th><th>Model</th><th>Kategoria</th><th>Odpowiedź modelu</th><th>Composite</th><th>Hallucination</th>
  </tr></thead>
  <tbody id="drill-body"></tbody>
</table>
<div class="pagination">
  <button id="prev-btn">&#8592;</button>
  <span id="page-info"></span>
  <button id="next-btn">&#8594;</button>
</div>

<script>
const DATA = {data_json};

// ── Overview table ────────────────────────────────────────────────────────────
(function() {{
  const cols = ["avg_composite","avg_faithfulness","avg_latency_ms","avg_ttft_ms","best_trial_score"];
  const bests = {{}};
  cols.forEach(c => {{
    const vals = DATA.overview.map(r => r[c]).filter(v => v != null && typeof v === "number");
    if (!vals.length) {{
      bests[c] = null;
      return;
    }}
    const lowerIsBetter = c === "avg_latency_ms" || c === "avg_ttft_ms";
    bests[c] = lowerIsBetter ? Math.min(...vals) : Math.max(...vals);
  }});
  const tbody = document.getElementById("overview-body");
  DATA.overview.forEach(row => {{
    const ttftCell = row.avg_ttft_ms == null ? "—" : row.avg_ttft_ms.toFixed(0);
    const ttftBest = row.avg_ttft_ms != null && bests.avg_ttft_ms != null && row.avg_ttft_ms === bests.avg_ttft_ms;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${{row.model}}</td>
      <td class="${{row.avg_composite === bests.avg_composite ? 'best' : ''}}">${{row.avg_composite.toFixed(4)}}</td>
      <td class="${{row.avg_faithfulness === bests.avg_faithfulness ? 'best' : ''}}">${{row.avg_faithfulness.toFixed(4)}}</td>
      <td class="${{row.avg_latency_ms === bests.avg_latency_ms ? 'best' : ''}}">${{row.avg_latency_ms.toFixed(0)}}</td>
      <td class="${{ttftBest ? 'best' : ''}}">${{ttftCell}}</td>
      <td class="${{row.best_trial_score === bests.best_trial_score ? 'best' : ''}}">${{row.best_trial_score.toFixed(4)}}</td>
    `;
    tbody.appendChild(tr);
  }});
}})();

// ── Latency box plot (via bar + error bars approximation) ─────────────────────
(function() {{
  const models = Object.keys(DATA.latency_box);
  const medians = models.map(m => DATA.latency_box[m].median);
  const mins    = models.map(m => DATA.latency_box[m].min);
  const maxs    = models.map(m => DATA.latency_box[m].max);
  const p25s    = models.map(m => DATA.latency_box[m].p25);
  const p75s    = models.map(m => DATA.latency_box[m].p75);

  new Chart(document.getElementById("latencyChart"), {{
    type: "bar",
    data: {{
      labels: models,
      datasets: [
        {{
          label: "Median E2E (ms)",
          data: medians,
          backgroundColor: "rgba(52,152,219,0.7)",
        }},
        {{
          label: "P25",
          data: p25s,
          backgroundColor: "rgba(52,152,219,0.3)",
          type: "bar",
        }},
        {{
          label: "P75",
          data: p75s,
          backgroundColor: "rgba(52,152,219,0.15)",
          type: "bar",
        }},
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ position: "top" }},
        annotation: {{}}
      }},
      scales: {{
        y: {{
          title: {{ display: true, text: "ms" }},
          suggestedMax: 90000,
        }}
      }}
    }}
  }});
}})();

// ── Quality grouped bar ───────────────────────────────────────────────────────
(function() {{
  const models = Object.keys(DATA.quality_bars);
  const metrics = ["faithfulness","answer_relevancy","conciseness","polish_quality"];
  const colors = ["#2ecc71","#3498db","#9b59b6","#e67e22"];
  const datasets = metrics.map((m, i) => ({{
    label: m,
    data: models.map(mdl => DATA.quality_bars[mdl][m]),
    backgroundColor: colors[i],
  }}));
  new Chart(document.getElementById("qualityChart"), {{
    type: "bar",
    data: {{ labels: models, datasets }},
    options: {{
      responsive: true,
      scales: {{ y: {{ min: 0, max: 1 }} }},
      plugins: {{ legend: {{ position: "top" }} }}
    }}
  }});
}})();

// ── Scatter trade-off ─────────────────────────────────────────────────────────
(function() {{
  const pts = DATA.scatter.map(p => ({{ x: p.avg_e2e_ms, y: p.avg_faithfulness, label: p.model }}));
  new Chart(document.getElementById("scatterChart"), {{
    type: "scatter",
    data: {{
      datasets: [{{
        label: "Modele",
        data: pts,
        backgroundColor: "rgba(231,76,60,0.8)",
        pointRadius: 8,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => `${{ctx.raw.label}}: (${{ctx.raw.x.toFixed(0)}}ms, ${{ctx.raw.y.toFixed(3)}})`
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: "Avg E2E (ms)" }} }},
        y: {{ title: {{ display: true, text: "Avg Faithfulness" }}, min: 0, max: 1 }}
      }}
    }}
  }});
}})();

// ── Top params table ──────────────────────────────────────────────────────────
(function() {{
  const tbody = document.getElementById("params-body");
  Object.values(DATA.top_params).flat().forEach(row => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${{row.model}}</td>
      <td>${{row.trial}}</td>
      <td>${{row.temperature.toFixed(3)}}</td>
      <td>${{row.num_ctx}}</td>
      <td>${{row.num_predict}}</td>
      <td>${{row.composite_score.toFixed(4)}}</td>
    `;
    tbody.appendChild(tr);
  }});
}})();

// ── Drilldown table ───────────────────────────────────────────────────────────
(function() {{
  const PAGE_SIZE = 50;
  let page = 0;
  let filtered = [];

  const fModel = document.getElementById("f-model");
  const fCat   = document.getElementById("f-cat");
  const fHall  = document.getElementById("f-hall");

  // Populate filters
  const models = [...new Set(DATA.records.map(r => r.model))].sort();
  const cats   = [...new Set(DATA.records.map(r => r.category))].sort();
  models.forEach(m => fModel.insertAdjacentHTML("beforeend", `<option value="${{m}}">${{m}}</option>`));
  cats.forEach(c => fCat.insertAdjacentHTML("beforeend", `<option value="${{c}}">${{c}}</option>`));

  function applyFilters() {{
    const vm = fModel.value, vc = fCat.value, vh = fHall.value;
    filtered = DATA.records.filter(r =>
      (!vm || r.model === vm) &&
      (!vc || r.category === vc) &&
      (!vh || String(r.hallucination_flag) === vh)
    );
    page = 0;
    render();
  }}

  function render() {{
    const tbody = document.getElementById("drill-body");
    tbody.innerHTML = "";
    const slice = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
    slice.forEach(r => {{
      const tr = document.createElement("tr");
      const hall = r.hallucination_flag;
      tr.innerHTML = `
        <td>${{r.case_id}}</td>
        <td>${{r.model}}</td>
        <td>${{r.category}}</td>
        <td><span class="trunc" title="${{(r.model_answer || '').replace(/"/g,'&quot;')}}">${{r.model_answer || ''}}</span></td>
        <td>${{r.composite_score.toFixed(4)}}</td>
        <td class="tag-${{hall}}">${{hall}}</td>
      `;
      tbody.appendChild(tr);
    }});
    const total = Math.ceil(filtered.length / PAGE_SIZE) || 1;
    document.getElementById("page-info").textContent = `Strona ${{page+1}} / ${{total}} (${{filtered.length}} rekordów)`;
  }}

  document.getElementById("prev-btn").addEventListener("click", () => {{ if (page > 0) {{ page--; render(); }} }});
  document.getElementById("next-btn").addEventListener("click", () => {{
    if ((page + 1) * PAGE_SIZE < filtered.length) {{ page++; render(); }}
  }});
  [fModel, fCat, fHall].forEach(el => el.addEventListener("change", applyFilters));

  filtered = DATA.records;
  render();
}})();
</script>
</body>
</html>
"""


def generate_report(results_dir: str, output_path: str) -> None:
    records = _load_records(results_dir)
    if not records:
        print(f"Brak rekordów w {results_dir} — raport nie zostanie wygenerowany.")
        return

    agg = _aggregate(records)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    data_json = json.dumps(agg, ensure_ascii=False)
    html = _HTML_TEMPLATE.format(
        timestamp=timestamp,
        n_records=len(records),
        n_models=len(agg["overview"]),
        data_json=data_json,
    )
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Raport wygenerowany: {output_path} ({len(records)} rekordów, {len(agg['overview'])} modeli)")


if __name__ == "__main__":
    base = Path(__file__).parent
    generate_report(str(base / "results"), str(base / "report.html"))
