# Warm-cache n=100 — werdykt Gaussa

Wygenerowane przez `benchmark/analysis/run_warm_n100_diagnostics.py`.

- Dane n=100: `/Users/krzysztof/Documents/studia/magisterka/benchmark/results/quant_latency_warm_n100.jsonl`
- Dane n=20 (porównanie): `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/quant-warm-cache/warm_cache_generation_full_20260618_20260621.jsonl`
- Artefakty: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/warm-n100`

## Teza

Przy n=20 warm `generation_ms` padał na GAUSS_FAIL; hipoteza: większe n (100) upodobni rozkład do Optuna E2E (GAUSS_OK).

## Reguła werdyktu (bez zmian)

- `GAUSS_FAIL` — gdy `średnia − 2σ < 0` **lub** `|skew| > 1` **lub** `mean/median > 1.15`.
- `GAUSS_OK` — w przeciwnym razie.

## Wynik n=100

| metryka | GAUSS_OK | GAUSS_FAIL |
|---|---:|---:|
| `generation_ms` | 0 | 12 |
| `ttft_ms` | 10 | 2 |
| `e2e_ms` | 1 | 11 |

### generation_ms (kluczowe dla Tab. 5.7 / Rys. 5.7–5.10)

| model | wariant | n | mean [s] | median [s] | σ [s] | mean−2σ | skew | mean/med | werdykt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `bielik-4.5b` | Q2_K | 100 | 30.45 | 29.83 | 3.45 | 23.54 | -6.65 | 1.021 | **GAUSS_FAIL** |
| `bielik-4.5b` | Q4_0 | 100 | 34.14 | 34.53 | 2.85 | 28.44 | -2.55 | 0.989 | **GAUSS_FAIL** |
| `bielik-4.5b` | Q4_K_M | 100 | 35.13 | 36.22 | 5.48 | 24.17 | -3.69 | 0.970 | **GAUSS_FAIL** |
| `bielik-4.5b` | Q8_0 | 100 | 66.22 | 65.85 | 0.88 | 64.46 | 1.28 | 1.006 | **GAUSS_FAIL** |
| `gemma3-4b` | Q2_K | 100 | 19.18 | 28.59 | 11.70 | -4.21 | -0.27 | 0.671 | **GAUSS_FAIL** |
| `gemma3-4b` | Q4_0 | 100 | 6.18 | 4.98 | 3.96 | -1.75 | 2.62 | 1.241 | **GAUSS_FAIL** |
| `gemma3-4b` | Q4_K_M | 100 | 7.59 | 5.87 | 5.22 | -2.86 | 1.80 | 1.293 | **GAUSS_FAIL** |
| `gemma3-4b` | Q8_0 | 100 | 10.98 | 8.82 | 7.09 | -3.20 | 2.65 | 1.245 | **GAUSS_FAIL** |
| `gemma4-e2b` | Q2_K | 47 | 14.68 | 17.90 | 6.74 | 1.20 | -1.27 | 0.820 | **GAUSS_FAIL** |
| `gemma4-e2b` | Q4_0 | 100 | 6.08 | 4.39 | 5.52 | -4.96 | 2.01 | 1.384 | **GAUSS_FAIL** |
| `gemma4-e2b` | Q4_K_M | 100 | 5.38 | 3.48 | 4.38 | -3.37 | 1.89 | 1.548 | **GAUSS_FAIL** |
| `gemma4-e2b` | Q8_0 | 100 | 7.24 | 5.71 | 4.89 | -2.53 | 2.19 | 1.269 | **GAUSS_FAIL** |

### TTFT

| model | wariant | n | mean [s] | σ [s] | skew | mean/med | werdykt |
|---|---|---:|---:|---:|---:|---:|---|
| `bielik-4.5b` | Q2_K | 100 | 8.78 | 3.09 | 9.08 | 1.055 | **GAUSS_FAIL** |
| `bielik-4.5b` | Q4_0 | 100 | 7.61 | 0.56 | 0.63 | 1.012 | **GAUSS_OK** |
| `bielik-4.5b` | Q4_K_M | 100 | 7.09 | 0.59 | 0.69 | 1.012 | **GAUSS_OK** |
| `bielik-4.5b` | Q8_0 | 100 | 6.56 | 0.49 | 0.40 | 1.014 | **GAUSS_OK** |
| `gemma3-4b` | Q2_K | 100 | 2.77 | 0.47 | 0.54 | 1.062 | **GAUSS_OK** |
| `gemma3-4b` | Q4_0 | 100 | 2.56 | 0.42 | 0.64 | 1.064 | **GAUSS_OK** |
| `gemma3-4b` | Q4_K_M | 100 | 2.47 | 0.41 | 0.50 | 1.022 | **GAUSS_OK** |
| `gemma3-4b` | Q8_0 | 100 | 2.54 | 0.37 | 0.41 | 1.039 | **GAUSS_OK** |
| `gemma4-e2b` | Q2_K | 47 | 3.71 | 3.48 | 3.16 | 1.442 | **GAUSS_FAIL** |
| `gemma4-e2b` | Q4_0 | 100 | 1.95 | 0.28 | 0.58 | 1.049 | **GAUSS_OK** |
| `gemma4-e2b` | Q4_K_M | 100 | 1.85 | 0.26 | 0.48 | 1.045 | **GAUSS_OK** |
| `gemma4-e2b` | Q8_0 | 100 | 1.89 | 0.24 | 0.53 | 1.039 | **GAUSS_OK** |

## Porównanie n=20 vs n=100 (`generation_ms`)

| model | wariant | n20→n100 | skew 20→100 | mean/med 20→100 | werdykt 20 | werdykt 100 | zmiana? |
|---|---|---:|---|---|---|---|---|
| `bielik-4.5b` | Q2_K | 20→100 | 1.02→-6.65 | 1.029→1.021 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `bielik-4.5b` | Q4_0 | 20→100 | -2.50→-2.55 | 1.004→0.989 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `bielik-4.5b` | Q4_K_M | 20→100 | -3.65→-3.69 | 0.967→0.970 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `bielik-4.5b` | Q8_0 | 20→100 | 1.06→1.28 | 1.007→1.006 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `gemma3-4b` | Q2_K | 20→100 | -0.57→-0.27 | 0.721→0.671 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `gemma3-4b` | Q4_0 | 20→100 | 2.23→2.62 | 1.105→1.241 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `gemma3-4b` | Q4_K_M | 20→100 | 1.91→1.80 | 1.265→1.293 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `gemma3-4b` | Q8_0 | 20→100 | 2.23→2.65 | 1.271→1.245 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `gemma4-e2b` | Q2_K | 9→47 | -1.88→-1.27 | 0.929→0.820 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `gemma4-e2b` | Q4_0 | 20→100 | 1.59→2.01 | 1.461→1.384 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `gemma4-e2b` | Q4_K_M | 20→100 | 1.70→1.89 | 1.584→1.548 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |
| `gemma4-e2b` | Q8_0 | 20→100 | 1.96→2.19 | 1.263→1.269 | **GAUSS_FAIL** | **GAUSS_FAIL** | nie |

## Wniosek

Żadna grupa `generation_ms` **nie zmieniła** werdyktu FAIL→OK przy przejściu n=20→100. Teza „więcej case’ów naprawi Gaussa” **nie potwierdza się** dla warm generation.

- Warm `generation_ms` przy n=100: nadal **strukturalnie skośny** (ogon odpowiedzi / wariancja długości). Nie raportować mean±2σ jako „95% pomiarów”.
- Warm `ttft_ms`: nadal zbliżony do Gaussa — **mean ± σ** obronne.
- Optuna Optimization Full E2E pozostaje osobną historią (GAUSS_OK przy dużym n) — inny eksperyment / inna metryka.

## Rekomendacja do TeXu (bez regeneracji teraz)

| Miejsce | Estymator |
|---|---|
| Tab. 5.7 / Rys. 5.7–5.8–5.10 generation | **median + MAD** (lub p50/p90) |
| Rys. 5.12 TTFT warm | **mean ± σ** |

### Rekomendacje z `continuous_summary` (n=100)

| model/wariant | generation family | location | scale |
|---|---|---|---|
| `bielik-4.5b/Q2_K` | skewed_robust | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `bielik-4.5b/Q4_0` | skewed_robust | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `bielik-4.5b/Q4_K_M` | skewed_robust | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `bielik-4.5b/Q8_0` | skewed_consider_lognormal | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma3-4b/Q2_K` | skewed_robust | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma3-4b/Q4_0` | skewed_consider_lognormal | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma3-4b/Q4_K_M` | skewed_consider_lognormal | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma3-4b/Q8_0` | skewed_consider_lognormal | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma4-e2b/Q2_K` | skewed_robust | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma4-e2b/Q4_0` | skewed_consider_lognormal | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma4-e2b/Q4_K_M` | skewed_consider_lognormal | median | MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma4-e2b/Q8_0` | skewed_consider_lognormal | median | MAD (→ σ_equiv = 1.4826·MAD) |

## Uwaga o danych

- `gemma4-e2b/Q2_K` generation: n=47 (mniej niż 100 ważnych `generation_ms` — część rekordów bez TTFT/gen).

## Gdzie patrzeć

- Arkusze: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/warm-n100/latency-dist/sheets/`
- Grupy 3-panel: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/warm-n100/latency-dist/groups/`
- QQ: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/warm-n100/figures/`
- CSV: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/warm-n100/tables/warm_n100_latency.csv`, `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/warm-n100/tables/warm_n20_vs_n100.csv`
