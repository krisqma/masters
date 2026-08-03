# Cold Gemmy — model LogNormal (na dostępnych danych)

Źródło: `/Users/krzysztof/Documents/studia/magisterka/benchmark/results/quant_latency_cold_n100.jsonl`
Artefakty: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/cold-gemma-lognormal`

Bielik pominięty. Placeholdery gdy `n_valid < 30`.

## Status komórek

| model | wariant | metryka | n_rows | n_valid | status |
|---|---|---|---:|---:|---|
| `gemma3-4b` | Q2_K | `e2e_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q2_K | `generation_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q2_K | `ttft_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q4_0 | `e2e_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q4_0 | `generation_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q4_0 | `ttft_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q4_K_M | `e2e_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q4_K_M | `generation_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q4_K_M | `ttft_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q8_0 | `e2e_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q8_0 | `generation_ms` | 100 | 100 | ok |
| `gemma3-4b` | Q8_0 | `ttft_ms` | 100 | 100 | ok |
| `gemma4-e2b` | Q2_K | `e2e_ms` | 100 | 100 | ok |
| `gemma4-e2b` | Q2_K | `generation_ms` | 100 | 43 | partial |
| `gemma4-e2b` | Q2_K | `ttft_ms` | 100 | 43 | partial |
| `gemma4-e2b` | Q4_0 | `e2e_ms` | 100 | 100 | ok |
| `gemma4-e2b` | Q4_0 | `generation_ms` | 100 | 100 | ok |
| `gemma4-e2b` | Q4_0 | `ttft_ms` | 100 | 100 | ok |
| `gemma4-e2b` | Q4_K_M | `e2e_ms` | 100 | 100 | ok |
| `gemma4-e2b` | Q4_K_M | `generation_ms` | 100 | 100 | ok |
| `gemma4-e2b` | Q4_K_M | `ttft_ms` | 100 | 100 | ok |
| `gemma4-e2b` | Q8_0 | `e2e_ms` | 64 | 64 | partial |
| `gemma4-e2b` | Q8_0 | `generation_ms` | 64 | 64 | partial |
| `gemma4-e2b` | Q8_0 | `ttft_ms` | 64 | 64 | partial |

## Tendencja: Normal vs LogNormal (`generation_ms`, n_valid≥30)

Preferencje AIC: {'lognormal': 6, 'normal': 2}

| model | wariant | n | μ_log | σ_log | e^μ [s] | e^{μ±2σ} [s] | AIC_N | AIC_LN | prefer |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `gemma3-4b` | Q2_K | 100 | 2.563 | 0.911 | 12.98 | [2.10, 80.24] | 777 | 781 | **normal** |
| `gemma3-4b` | Q4_0 | 100 | 1.604 | 0.443 | 4.97 | [2.05, 12.05] | 529 | 445 | **lognormal** |
| `gemma3-4b` | Q4_K_M | 100 | 1.762 | 0.523 | 5.82 | [2.05, 16.58] | 593 | 510 | **lognormal** |
| `gemma3-4b` | Q8_0 | 100 | 2.188 | 0.463 | 8.91 | [3.53, 22.48] | 667 | 570 | **lognormal** |
| `gemma4-e2b` | Q2_K | 43 | 2.387 | 1.244 | 10.88 | [0.90, 130.85] | 286 | 349 | **normal** |
| `gemma4-e2b` | Q4_0 | 100 | 1.400 | 0.674 | 4.06 | [1.05, 15.61] | 609 | 488 | **lognormal** |
| `gemma4-e2b` | Q4_K_M | 100 | 1.406 | 0.659 | 4.08 | [1.09, 15.25] | 577 | 485 | **lognormal** |
| `gemma4-e2b` | Q8_0 | 64 | 1.670 | 0.507 | 5.31 | [1.93, 14.63] | 370 | 311 | **lognormal** |

## Wniosek (tendencja)

Spośród 8 komórek generation z pełnym fittem: LogNormal wygrywa AIC w **6**, Normal w **2**, remis **0**.

- Cold generation Gemm: trzymaj **LogNormal** jako model roboczy (pas \(e^{\hat\mu\pm 2\hat\sigma}\)).
- Normal na sekundach często daje ujemne \(\bar x-2s\) albo gorszy AIC.
- Gemma4 Q8_0 / częściowe komórki: placeholdery — odpal skrypt ponownie po dograniu.

## Gdzie patrzeć

- Arkusze: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/cold-gemma-lognormal/latency-dist/sheets/`
- Grupy: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/cold-gemma-lognormal/latency-dist/groups/`
- QQ log: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/cold-gemma-lognormal/figures/`
- CSV: `/Users/krzysztof/Documents/studia/magisterka/benchmark/analysis/stat-model-diagnostics/cold-gemma-lognormal/tables/`
