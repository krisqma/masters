# Review figures — rozdział 5 (LN / Gauss)

Wygenerowane przez `generate_review_figures_ch5.py`. **Nie są jeszcze wpięte w TeX.**

## Polityka

| Folder | Estymator |
|---|---|
| `warm/`, `cold/` | **Log-normal**: wysokość słupka = mediana modelu LN (e^μ); przedział = e^(μ±σ) |
| `optuna/` jakość | bez zmian (mean / rate) |
| `optuna/04_latency_hallucination_pareto_selected` | **Gauss**: mean E2E ± σ |

Nomenklatura: **nie** „centrum” / moda PDF — wysokość to **mediana modelu LN** (e^μ).

## Matryce dopasowania (`fit/`)

- `warm_generation_ln_fit_matrix` — hist generation + krzywa **LN** (3×4)
- `cold_generation_ln_fit_matrix` — to samo, cold
- `optuna_e2e_gauss_fit_matrix` — hist E2E + krzywa **Normal** (4 modele Optuny)

## Dane

- Warm: `/Users/krzysztof/Documents/studia/magisterka/benchmark/results/quant_latency_warm_n100.jsonl`
- Cold: `/Users/krzysztof/Documents/studia/magisterka/benchmark/results/quant_latency_cold_n100.jsonl`
- Optuna: `benchmark/analysis/results-*/` (głównie `results-selected-9.05`)

## Mapa: stary plik w TeXu → nowy

| Stary stem (wykresy/) | Nowy (warm/ lub optuna/) | Estymator |
|---|---|---|
| `01_mediana_generacji_wedlug_wariantu` | `warm/01_generation_wedlug_wariantu` | LN |
| `02_mediana_ttft_wedlug_wariantu` | `warm/02_ttft_wedlug_wariantu` | LN |
| `03_dekompozycja_e2e` | `warm/03_dekompozycja_e2e` | LN |
| `04_przyspieszenie_wzgledem_q8` | `warm/04_przyspieszenie_wzgledem_q8` | LN speedup |
| `05_generacja_a_dlugosc_odpowiedzi` | `warm/05_generacja_a_dlugosc_odpowiedzi` | scatter |
| `06_rozmiar_gguf_a_generacja` | `warm/06_rozmiar_gguf_a_generacja` | LN |
| `07_mediana_generacji_ms_wedlug_wariantu` | `warm/07_generation_ms_wedlug_wariantu` | LN |
| `08_szybkosc_generacji_tok_s_wedlug_wariantu` | `warm/08_szybkosc_generacji_tok_s_wedlug_wariantu` | LN |
| `02_score_vs_hallucination_selected` | `optuna/02_...` | jakość |
| `03_nonhall_score_spread_selected` | `optuna/03_...` | jakość |
| `04_latency_hallucination_pareto_selected` | `optuna/04_...` | **Gauss mean±σ** |
| `05_case_hallucination_rates_selected` | `optuna/05_...` | jakość |
| `06_hparam_correlation_heatmap_selected` | `optuna/06_...` | jakość |
| `07_regression_r2_comparison_selected` | `optuna/07_...` | jakość |

Cold: te same stemy z prefixem `cold_` w `cold/`.

## Tabele

- `tables/warm_ln_summary.csv`, `tables/cold_ln_summary.csv` — μ, σ, mediana LN e^μ, e^(μ±σ), speedup
- `tables/optuna_selected_gauss_e2e.csv` — mean/σ E2E per model

## Podpisy

Propozycje LaTeX: `CAPTIONS.tex` (nie includowane w `main.tex`).
