# Werdykt wizualny: czy latencja to Gauss?

Wygenerowane przez `benchmark/analysis/latency_distribution_plots.py`.

## Legenda osi

- **Panel A:** X = numer pytania, Y = czas [s] (surowe punkty).
- **Panel B:** X = czas [s], Y = liczność w binie (histogram) + krzywe Gauss / log-normal / KDE.
- **Panel C:** porównanie pasm mean±σ z percentylami p50/p90/p95.

## Reguła werdyktu

- `GAUSS_FAIL` — gdy `średnia − 2σ < 0` **lub** `|skew| > 1` **lub** `mean/median > 1.15`.
- `GAUSS_OK` — w przeciwnym razie (Gauss jako model roboczy do mean±σ).

## Tabela

| eksperyment | model | wariant | metryka | n | mean [s] | σ [s] | mean−2σ [s] | p50 | p90 | skew | mean/med | werdykt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Optuna Optimization Full (cold start) | `gemma2:2b` | — | `e2e_ms` | 600 | 20.53 | 5.18 | 10.18 | 20.87 | 27.03 | -0.20 | 0.984 | **GAUSS_OK** |
| Optuna Optimization Full (cold start) | `gemma3:4b` | — | `e2e_ms` | 600 | 30.40 | 7.56 | 15.27 | 31.41 | 41.16 | 0.12 | 0.968 | **GAUSS_OK** |
| Optuna Optimization Full (cold start) | `llama3.2:3b` | — | `e2e_ms` | 200 | 26.80 | 6.30 | 14.19 | 27.56 | 34.56 | -0.14 | 0.972 | **GAUSS_OK** |
| Optuna Optimization Full (cold start) | `qwen2.5:3b` | — | `e2e_ms` | 200 | 29.38 | 6.62 | 16.13 | 30.19 | 37.53 | 0.02 | 0.973 | **GAUSS_OK** |
| warm-cache | `bielik-4.5b` | Q2_K | `e2e_ms` | 20 | 39.63 | 2.02 | 35.60 | 38.93 | 43.42 | 0.84 | 1.018 | **GAUSS_OK** |
| warm-cache | `bielik-4.5b` | Q4_0 | `e2e_ms` | 20 | 39.55 | 2.81 | 33.93 | 39.44 | 41.51 | -2.37 | 1.003 | **GAUSS_FAIL** |
| warm-cache | `bielik-4.5b` | Q4_K_M | `e2e_ms` | 20 | 40.64 | 4.92 | 30.80 | 41.68 | 42.93 | -3.45 | 0.975 | **GAUSS_FAIL** |
| warm-cache | `bielik-4.5b` | Q8_0 | `e2e_ms` | 20 | 72.64 | 0.95 | 70.74 | 72.48 | 73.96 | 1.12 | 1.002 | **GAUSS_FAIL** |
| warm-cache | `gemma3-4b` | Q2_K | `e2e_ms` | 20 | 25.00 | 12.24 | 0.51 | 33.44 | 35.12 | -0.56 | 0.748 | **GAUSS_OK** |
| warm-cache | `gemma3-4b` | Q4_0 | `e2e_ms` | 20 | 9.35 | 4.37 | 0.62 | 8.46 | 12.38 | 2.14 | 1.106 | **GAUSS_FAIL** |
| warm-cache | `gemma3-4b` | Q4_K_M | `e2e_ms` | 20 | 9.94 | 5.45 | -0.97 | 8.11 | 15.74 | 1.85 | 1.226 | **GAUSS_FAIL** |
| warm-cache | `gemma3-4b` | Q8_0 | `e2e_ms` | 20 | 13.53 | 7.15 | -0.76 | 10.83 | 20.81 | 2.21 | 1.250 | **GAUSS_FAIL** |
| warm-cache | `gemma4-e2b` | Q2_K | `e2e_ms` | 20 | 20.57 | 4.20 | 12.18 | 21.56 | 22.13 | -3.66 | 0.954 | **GAUSS_FAIL** |
| warm-cache | `gemma4-e2b` | Q4_0 | `e2e_ms` | 20 | 8.17 | 6.08 | -4.00 | 6.26 | 19.83 | 1.56 | 1.304 | **GAUSS_FAIL** |
| warm-cache | `gemma4-e2b` | Q4_K_M | `e2e_ms` | 20 | 7.25 | 4.65 | -2.04 | 5.26 | 11.84 | 1.67 | 1.377 | **GAUSS_FAIL** |
| warm-cache | `gemma4-e2b` | Q8_0 | `e2e_ms` | 20 | 9.05 | 5.01 | -0.97 | 7.30 | 14.40 | 1.93 | 1.240 | **GAUSS_FAIL** |
| warm-cache | `bielik-4.5b` | Q2_K | `generation_ms` | 20 | 31.00 | 1.81 | 27.38 | 30.14 | 33.83 | 1.02 | 1.029 | **GAUSS_FAIL** |
| warm-cache | `bielik-4.5b` | Q4_0 | `generation_ms` | 20 | 31.99 | 2.68 | 26.64 | 31.88 | 34.52 | -2.50 | 1.004 | **GAUSS_FAIL** |
| warm-cache | `bielik-4.5b` | Q4_K_M | `generation_ms` | 20 | 33.37 | 4.66 | 24.05 | 34.52 | 35.44 | -3.65 | 0.967 | **GAUSS_FAIL** |
| warm-cache | `bielik-4.5b` | Q8_0 | `generation_ms` | 20 | 65.99 | 0.79 | 64.42 | 65.55 | 67.01 | 1.06 | 1.007 | **GAUSS_FAIL** |
| warm-cache | `gemma3-4b` | Q2_K | `generation_ms` | 20 | 22.17 | 11.99 | -1.80 | 30.77 | 31.74 | -0.57 | 0.721 | **GAUSS_FAIL** |
| warm-cache | `gemma3-4b` | Q4_0 | `generation_ms` | 20 | 6.78 | 4.02 | -1.25 | 6.14 | 9.50 | 2.23 | 1.105 | **GAUSS_FAIL** |
| warm-cache | `gemma3-4b` | Q4_K_M | `generation_ms` | 20 | 7.47 | 5.13 | -2.80 | 5.90 | 12.29 | 1.91 | 1.265 | **GAUSS_FAIL** |
| warm-cache | `gemma3-4b` | Q8_0 | `generation_ms` | 20 | 10.98 | 6.90 | -2.82 | 8.64 | 17.76 | 2.23 | 1.271 | **GAUSS_FAIL** |
| warm-cache | `gemma4-e2b` | Q2_K | `generation_ms` | 9 | 15.98 | 5.95 | 4.09 | 17.20 | 19.45 | -1.88 | 0.929 | **GAUSS_FAIL** |
| warm-cache | `gemma4-e2b` | Q4_0 | `generation_ms` | 20 | 6.24 | 5.89 | -5.54 | 4.27 | 17.83 | 1.59 | 1.461 | **GAUSS_FAIL** |
| warm-cache | `gemma4-e2b` | Q4_K_M | `generation_ms` | 20 | 5.39 | 4.50 | -3.62 | 3.40 | 9.76 | 1.70 | 1.584 | **GAUSS_FAIL** |
| warm-cache | `gemma4-e2b` | Q8_0 | `generation_ms` | 20 | 7.18 | 4.86 | -2.55 | 5.68 | 12.42 | 1.96 | 1.263 | **GAUSS_FAIL** |
| warm-cache | `bielik-4.5b` | Q2_K | `ttft_ms` | 20 | 8.63 | 0.73 | 7.18 | 8.53 | 9.61 | 0.87 | 1.012 | **GAUSS_OK** |
| warm-cache | `bielik-4.5b` | Q4_0 | `ttft_ms` | 20 | 7.55 | 0.57 | 6.41 | 7.55 | 8.28 | 0.64 | 1.000 | **GAUSS_OK** |
| warm-cache | `bielik-4.5b` | Q4_K_M | `ttft_ms` | 20 | 7.26 | 0.62 | 6.02 | 7.20 | 7.99 | 0.35 | 1.009 | **GAUSS_OK** |
| warm-cache | `bielik-4.5b` | Q8_0 | `ttft_ms` | 20 | 6.65 | 0.50 | 5.65 | 6.52 | 7.35 | 0.41 | 1.020 | **GAUSS_OK** |
| warm-cache | `gemma3-4b` | Q2_K | `ttft_ms` | 20 | 2.83 | 0.47 | 1.89 | 2.74 | 3.45 | 0.21 | 1.030 | **GAUSS_OK** |
| warm-cache | `gemma3-4b` | Q4_0 | `ttft_ms` | 20 | 2.57 | 0.42 | 1.72 | 2.42 | 3.06 | 0.59 | 1.061 | **GAUSS_OK** |
| warm-cache | `gemma3-4b` | Q4_K_M | `ttft_ms` | 20 | 2.47 | 0.43 | 1.61 | 2.38 | 2.86 | 0.77 | 1.038 | **GAUSS_OK** |
| warm-cache | `gemma3-4b` | Q8_0 | `ttft_ms` | 20 | 2.55 | 0.39 | 1.77 | 2.48 | 2.95 | 0.31 | 1.028 | **GAUSS_OK** |
| warm-cache | `gemma4-e2b` | Q2_K | `ttft_ms` | 9 | 3.28 | 1.32 | 0.64 | 3.17 | 4.67 | 0.49 | 1.032 | **GAUSS_OK** |
| warm-cache | `gemma4-e2b` | Q4_0 | `ttft_ms` | 20 | 1.92 | 0.31 | 1.31 | 1.84 | 2.25 | 0.49 | 1.044 | **GAUSS_OK** |
| warm-cache | `gemma4-e2b` | Q4_K_M | `ttft_ms` | 20 | 1.86 | 0.25 | 1.37 | 1.88 | 2.12 | 0.08 | 0.989 | **GAUSS_OK** |
| warm-cache | `gemma4-e2b` | Q8_0 | `ttft_ms` | 20 | 1.88 | 0.23 | 1.42 | 1.86 | 2.16 | 0.21 | 1.011 | **GAUSS_OK** |

**Podsumowanie:** GAUSS_OK = 18, GAUSS_FAIL = 22 (z 40 grup).

## Gdzie patrzeć w pierwszej kolejności

- Arkusz warm generation (najczęściej FAIL): `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/latency-dist/sheets/sheet_warm_gemma3_generation.png`
- Arkusz Optuna Optimization Full (cold start) E2E: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/latency-dist/sheets/sheet_optuna_optimization_full_e2e.png`
- Wszystkie grupy 3-panelowe: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/latency-dist/groups/`

## Rekomendacja do pracy

- Warm-cache `generation_ms`: **12/12** grup = GAUSS_FAIL → do raportowania lepiej **p50 / p90** (albo mean z zastrzeżeniem o ogonie), nie mean±2σ jako „95% pomiarów”.
- Optuna Optimization Full (cold start) `e2e_ms`: **4/4** modeli = GAUSS_OK → mean±σ jest obronne jako model roboczy.

