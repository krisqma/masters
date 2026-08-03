# Werdykt modelu statystycznego — rozdział 5

Wygenerowane przez `benchmark/analysis/stat_model_diagnostics.py`.
Bez regeneracji wykresów TeX — tylko diagnostyka i decyzja modelowa.

## 1. Mapa figura/tabela → dane → skrypt

| Artefakt | Dane | Producer | Estymator dziś |
|---|---|---|---|
| `tab:ranking_run1 (Tab. 5.4)` | `results-throttling/*.jsonl` | `thesis_statistical_analysis.model_summary` | score=mean; e2e=median; temp=median w tekście |
| `tab:ranking_run2 (Tab. 5.5)` | `results-5-trials-nothrottling/*.jsonl` | `thesis_statistical_analysis.model_summary` | score=mean; e2e=median |
| `fig:score_vs_hall (Rys. 5.2)` | `results-selected-9.05/*.jsonl` | `thesis_statistical_analysis.make_figures / trial_summary` | punkt = mean score i hall_rate per trial Optuny |
| `fig:regression_r2 (Rys. 5.3)` | `results-selected-9.05/*.jsonl` | `thesis_statistical_analysis.regression_comparison` | R² z OLS / linregress na agregatach trial |
| `tab:kwantyzacja_wyniki (Tab. 5.7)` | `warm_cache_generation_full_*.jsonl` | `warm_cache_generation_analysis.build_tables` | TTFT/gen/E2E = median; speedup na medianach |
| `fig:dekompozycja_e2e (Rys. 5.7)` | `warm_cache_generation_full_*.jsonl` | `warm_cache_generation_analysis.plot_e2e_decomposition` | słupki z median TTFT + median generation |
| `fig:speedup_q8 (Rys. 5.8)` | `warm_cache_generation_full_*.jsonl` | `warm_cache_generation_analysis.plot_speedup` | speedup = median_gen(Q8) / median_gen(wariant) |
| `fig:tok_s_variant (Rys. 5.10)` | `warm_cache_generation_full_*.jsonl` | `warm_cache_generation_analysis (chars/s proxy, potem median)` | median(answer_chars / (generation_ms/1000)) |
| `fig:ttft_warm (Rys. 5.12)` | `warm_cache_generation_full_*.jsonl` | `warm_cache_generation_analysis.grouped_bar TTFT` | median TTFT |
| `tab:wyniki_finalne (Tab. 5.8)` | `results-selected-9.05 → selected_model_confidence_intervals.csv` | `thesis_statistical_analysis.selected_model_ci / bootstrap_ci` | score/hall = mean + bootstrap CI α=0.05 (kod n_boot=5000; TeX pisze 10000); E2E = median bez CI |

## 2. Język σ / SE / „95%” (ujednolicenie)

| Pasmo | Pokrycie przy \(N(\mu,\sigma^2)\) | Co oznacza w pracy |
|---|---|---|
| ±1σ | ≈ 68,3% | rozrzut **pojedynczych** pomiarów wokół średniej |
| ±2σ | ≈ 95,4% | szeroki pas rozrzutu pomiarów (nie CI średniej) |
| ±3σ | ≈ 99,7% | prawie cały rozkład przy założeniu Gaussa |
| ±1,96·SE | ≈ 95% dla **estymatora** średniej/proporcji | to jest odpowiednik obecnego „95% CI” |

**Ważne rozróżnienie:**

- **σ** = odchylenie standardowe próbki (rozrzut pojedynczych rekordów).
- **SE = σ/√n** = błąd standardowy **średniej** (niepewność estymatora).
- Obecne „95% CI” z bootstrapa ≈ percentyle 2,5%/97,5% rozkładu **średnich bootstrapowych** ≈ ±1,96·SE przy CLT.
- To **nie** jest „3σ ≈ 96%”. Przy Gaussie ~95% to **~2σ**, a 3σ to ~99,7%.

### Bootstrap: kod vs TeX

- W kodzie (`thesis_statistical_analysis.bootstrap_ci`): **n_boot = 5000**.
- W TeXu przy Tab. 5.8: napisane **10000** prób.
- **Decyzja metodyczna:** w tekście podawać **5000** (zgodne z kodem),
  albo podnieść `n_boot` do 10 000 i przebudować CSV — jedna wartość, bez rozjazdu.
  Na razie w diagnostyce porównujemy bootstrap przy n_boot = 5000 z pasmem ±1,96·SE.

## 3. Optuna Optimization Full (cold start) — folder `selected-9.05` — E2E / score / hall

### 3.1 Latencja E2E

| model | n | mean | median | σ | mean/med | skew | model_family | rekomendacja |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `gemma2:2b` | 600 | 20528.9 | 20866.8 | 5176.0 | 0.984 | -0.20 | gaussian_raw | mean ± std (σ) |
| `gemma3:4b` | 600 | 30401.5 | 31407.7 | 7563.3 | 0.968 | 0.12 | gaussian_raw | mean ± std (σ) |
| `llama3.2:3b` | 200 | 26800.7 | 27563.4 | 6304.3 | 0.972 | -0.14 | gaussian_raw | mean ± std (σ) |
| `qwen2.5:3b` | 200 | 29375.1 | 30185.8 | 6621.7 | 0.973 | 0.02 | gaussian_raw | mean ± std (σ) |

**Werdykt E2E (Optuna Optimization Full, cold start):** rozkład prawie symetryczny (mean/med ≈ 0.97–0.98, |skew| małe). Mediana nie jest konieczna — przechodzimy na **mean ± σ** (rozrzut) oraz **mean ± 1,96·SE** (niepewność średniej).

### 3.2 Composite score

| model | n | mean | σ | SE | ±1.96·SE | bootstrap CI (5000) |
|---|---:|---:|---:|---:|---|---|
| `gemma2:2b` | 600 | 0.7315 | 0.3620 | 0.0148 | [0.7025, 0.7605] | [0.7019, 0.7601] |
| `gemma3:4b` | 600 | 0.8837 | 0.1342 | 0.0055 | [0.8730, 0.8945] | [0.8724, 0.8942] |
| `llama3.2:3b` | 200 | 0.7321 | 0.3480 | 0.0246 | [0.6838, 0.7803] | [0.6834, 0.7804] |
| `qwen2.5:3b` | 200 | 0.6967 | 0.3728 | 0.0264 | [0.6450, 0.7483] | [0.6453, 0.7460] |

**Werdykt score:** model **średniej z CLT** (bounded [0,1], atom w 0 przy hall). Raportować mean ± SE; „95%” = ±1,96·SE lub bootstrap mean (powinny być blisko).

### 3.3 Halucynacje (binomial)

| model | n | k | p̂ | SE_bin | ±1.96·SE | bootstrap CI (5000) |
|---|---:|---:|---:|---:|---|---|
| `gemma2:2b` | 600 | 110 | 0.1833 | 0.0158 | [0.1524, 0.2143] | [0.1533, 0.2150] |
| `gemma3:4b` | 600 | 7 | 0.0117 | 0.0044 | [0.0031, 0.0203] | [0.0050, 0.0217] |
| `llama3.2:3b` | 200 | 33 | 0.1650 | 0.0262 | [0.1136, 0.2164] | [0.1150, 0.2200] |
| `qwen2.5:3b` | 200 | 41 | 0.2050 | 0.0285 | [0.1490, 0.2610] | [0.1500, 0.2600] |

**Werdykt hall:** model **dwumianowy** \(X\sim\mathrm{Bin}(n,p)\). Dla gemma3:4b p jest małe (~1%) — SE binominalne i bootstrap zostają; nie mówić o „σ score’u” w kontekście odsetka halucynacji.

### 3.4 Jak przepisać Tab. 5.8 (język metodyczny)

Docelowy opis (bez wdrażania w TeX na tym etapie):

> Wyniki Optuna Optimization Full (cold start): średni composite score ± 1,96·SE (CLT / równoważnie bootstrap średniej, 
> n_boot=5000); odsetek halucynacji z SE binominalnym 
> \(\mathrm{{SE}}=\sqrt{{\hat p(1-\hat p)/n}}\); E2E jako średnia ± σ 
> (rozkład prawie symetryczny; mediana ≈ średnia).

Kolumna „E2E med.” → „E2E mean ± σ” (lub mean z SE, jeśli chodzi o niepewność średniej).
Nie mieszać w jednym wierszu CI średniej score z medianą E2E bez etykiety.

## 4. Warm-cache (kwantyzacja) — TTFT vs generation_ms

### 4.1 TTFT

| model/wariant | n | mean | median | σ | mean/med | skew | family | rekomendacja |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `bielik-4.5b/Q2_K` | 20 | 8631.9 | 8527.8 | 726.4 | 1.012 | 0.87 | skewed_consider_lognormal | median |
| `bielik-4.5b/Q4_0` | 20 | 7551.5 | 7553.4 | 568.9 | 1.000 | 0.64 | skewed_consider_lognormal | median |
| `bielik-4.5b/Q4_K_M` | 20 | 7261.7 | 7197.1 | 621.8 | 1.009 | 0.35 | gaussian_raw | mean |
| `bielik-4.5b/Q8_0` | 20 | 6652.3 | 6521.9 | 502.7 | 1.020 | 0.41 | gaussian_raw | mean |
| `gemma3-4b/Q2_K` | 20 | 2826.1 | 2743.7 | 468.2 | 1.030 | 0.21 | gaussian_raw | mean |
| `gemma3-4b/Q4_0` | 20 | 2569.0 | 2420.1 | 422.9 | 1.061 | 0.59 | skewed_consider_lognormal | median |
| `gemma3-4b/Q4_K_M` | 20 | 2469.0 | 2378.0 | 431.1 | 1.038 | 0.77 | skewed_consider_lognormal | median |
| `gemma3-4b/Q8_0` | 20 | 2548.6 | 2478.7 | 389.4 | 1.028 | 0.31 | gaussian_raw | mean |
| `gemma4-e2b/Q2_K` | 9 | 3277.5 | 3174.6 | 1320.3 | 1.032 | 0.49 | gaussian_raw | mean |
| `gemma4-e2b/Q4_0` | 20 | 1921.8 | 1840.2 | 305.4 | 1.044 | 0.49 | gaussian_raw | mean |
| `gemma4-e2b/Q4_K_M` | 20 | 1860.8 | 1881.2 | 247.5 | 0.989 | 0.08 | gaussian_raw | mean |
| `gemma4-e2b/Q8_0` | 20 | 1878.6 | 1857.8 | 229.3 | 1.011 | 0.21 | gaussian_raw | mean |

**Werdykt TTFT warm (Rys. 5.12):** niemal zawsze mean≈median, |skew| umiarkowany. Przejść na **mean ± σ**; mediana może zostać w tekście jako kontrola odporności, nie jako główny estymator.

### 4.2 generation_ms

| model/wariant | n | mean | median | σ | MAD·1.4826 | mean/med | skew | log_skew | family | rekomendacja |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `bielik-4.5b/Q2_K` | 20 | 30996.7 | 30137.3 | 1807.6 | 664.3 | 1.029 | 1.02 | 0.99 | skewed_consider_lognormal | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `bielik-4.5b/Q4_0` | 20 | 31994.4 | 31881.5 | 2676.7 | 363.1 | 1.004 | -2.50 | -2.89 | skewed_robust | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `bielik-4.5b/Q4_K_M` | 20 | 33374.0 | 34523.0 | 4662.0 | 1245.3 | 0.967 | -3.65 | -3.75 | skewed_robust | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `bielik-4.5b/Q8_0` | 20 | 65985.2 | 65554.9 | 785.0 | 490.7 | 1.007 | 1.06 | 1.03 | skewed_consider_lognormal | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma3-4b/Q2_K` | 20 | 22171.5 | 30767.8 | 11985.7 | 1581.7 | 0.721 | -0.57 | -0.87 | skewed_robust | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma3-4b/Q4_0` | 20 | 6784.8 | 6141.1 | 4019.6 | 2689.3 | 1.105 | 2.23 | 0.93 | skewed_consider_lognormal | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma3-4b/Q4_K_M` | 20 | 7467.1 | 5901.4 | 5131.2 | 2908.0 | 1.265 | 1.91 | 0.90 | skewed_consider_lognormal | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma3-4b/Q8_0` | 20 | 10980.6 | 8638.8 | 6899.9 | 3395.3 | 1.271 | 2.23 | 1.05 | skewed_consider_lognormal | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma4-e2b/Q2_K` | 9 | 15981.0 | 17203.8 | 5947.6 | 1342.9 | 0.929 | -1.88 | -2.06 | skewed_robust | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma4-e2b/Q4_0` | 20 | 6243.5 | 4272.5 | 5893.9 | 2442.4 | 1.461 | 1.59 | 0.93 | skewed_consider_lognormal | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma4-e2b/Q4_K_M` | 20 | 5388.5 | 3401.6 | 4503.7 | 1792.0 | 1.584 | 1.70 | 0.82 | skewed_consider_lognormal | median + MAD (→ σ_equiv = 1.4826·MAD) |
| `gemma4-e2b/Q8_0` | 20 | 7175.5 | 5683.1 | 4860.3 | 2965.3 | 1.263 | 1.96 | 0.79 | skewed_consider_lognormal | median + MAD (→ σ_equiv = 1.4826·MAD) |

**Werdykt generation_ms (Rys. 5.7/5.8/5.10, Tab. 5.7):** dla Gemm Q4/Q8 silny prawy ogon (mean/med często > 1.2, skew > 1). **Zostawić medianę + MAD** jako skalę odporną. Gdzie `log_skew` << `skew`, diagnostycznie rozważyć log-normal (QQ w `figures/qq_warm_loggen_*.png`) — bez zmiany PNG TeXu na tym etapie.

## 5. Run 1 / Run 2 (throttling) — E2E

### Run 1 (throttling)

| model | n | mean | median | σ | mean/med | skew | family |
|---|---:|---:|---:|---:|---:|---:|---|
| `gemma2:2b` | 100 | 36235.4 | 37229.3 | 9499.0 | 0.973 | -0.14 | gaussian_raw |
| `gemma3:4b` | 100 | 52998.0 | 57583.8 | 8645.6 | 0.920 | -0.94 | skewed_robust |
| `llama3.2:3b` | 100 | 46809.6 | 49198.8 | 11261.7 | 0.951 | -0.83 | skewed_robust |
| `phi3.5` | 100 | 59500.9 | 60000.0 | 2483.9 | 0.992 | -5.81 | skewed_robust |
| `qwen2.5:3b` | 100 | 49503.1 | 53169.1 | 9866.8 | 0.931 | -0.93 | skewed_robust |

### Run 2 (no throttling)

| model | n | mean | median | σ | mean/med | skew | family |
|---|---:|---:|---:|---:|---:|---:|---|
| `gemma2:2b` | 100 | 20429.2 | 20879.2 | 5163.5 | 0.978 | -0.20 | gaussian_raw |
| `gemma3:4b` | 100 | 31661.1 | 31742.3 | 8587.9 | 0.997 | 0.67 | skewed_consider_lognormal |
| `llama3.2:3b` | 100 | 26712.9 | 27144.8 | 6530.9 | 0.984 | 0.21 | gaussian_raw |
| `phi3.5` | 100 | 49178.2 | 52363.3 | 10904.0 | 0.939 | -0.93 | skewed_robust |
| `qwen2.5:3b` | 100 | 28997.3 | 29712.9 | 6659.0 | 0.976 | 0.00 | gaussian_raw |

Przy capie timeoutów / pustych odpowiedziach mediana bywa odporniejsza; po stabilizacji termicznej (Run 2/3) preferować mean±σ jeśli skew mały.

## 6. Podsumowanie decyzji (co zostaje / co zmienić później w TeXu)

| Miejsce | Estymator dziś | Decyzja modelowa |
|---|---|---|
| Tab. 5.8 score / hall | mean + „95% bootstrap” | mean / p̂ + **±1,96·SE** (σ języka: SE); bootstrap = weryfikacja; n_boot uzgodnić na **5000** |
| Tab. 5.8 E2E | median | **mean ± σ** (symetria); nie CI mediany |
| Rys. 5.12 TTFT warm | median | **mean ± σ** |
| Tab. 5.7 / Rys. 5.7–5.8–5.10 generation | median | **median + MAD** (skośność); opcjonalnie log-normal diagnostycznie |
| Rys. 5.2 / 5.3 | mean per trial / R² | bez zmian modelu; to agregaty jakości, nie latencja |
| Tab. 5.4 / 5.5 E2E | median | po diagnostyce Run1/2: mean±σ jeśli symetryczne, inaczej median+MAD |

## 7. Pliki wyjściowe

- `tables/inventory.csv`
- `tables/selected_latency.csv`
- `tables/selected_score.csv`
- `tables/selected_hallucination.csv`
- `tables/warm_latency.csv`
- `tables/run1_throttling_e2e.csv` / `tables/run2_nothrottling_e2e.csv` (jeśli źródła dostępne)
- `figures/qq_*.png` — diagnostyka normalności (nie do TeXu)

