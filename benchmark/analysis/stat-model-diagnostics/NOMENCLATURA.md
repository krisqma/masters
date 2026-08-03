# Nomenklatura eksperymentów (diagnostyka)

Żeby nie mylić folderów z tym, co było w głowie:

| Nazwa robocza | Folder na dysku | Co to było | Cache |
|---|---|---|---|
| **Optuna Optimization Full (cold start)** | `results-selected-9.05` / `selected-9.05` | Pełny run Optuny na 4 modelach (jakość + E2E/TTFT). Dużo rekordów, bo trial × 20 pytań. | **cold** — bez protokołu warm / prefix-cache |
| **Warm-cache kwantyzacja** | `quant-warm-cache/warm_cache_generation_*.jsonl` | Osobny eksperyment Q2/Q4/Q8; izolacja `generation_ms` | **warm** |
| Run 1 | `results-throttling` | Wczesna Optuna, throttling | cold / bez warm |
| Run 2 | `results-5-trials-nothrottling` | Optuna + chłodzenie | cold / bez warm |

Wykresy latencji:
- `latency-dist/sheets/sheet_optuna_optimization_full_e2e.png`
- `latency-dist/sheets/sheet_warm_gemma3_generation.png`
