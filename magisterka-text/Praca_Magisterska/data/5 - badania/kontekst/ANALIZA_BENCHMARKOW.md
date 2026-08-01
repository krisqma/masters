# Analiza wynikow benchmarkow

Ten katalog zawiera wyniki runow benchmarkowych zapisane jako pliki JSONL oraz skrypty do analizy statystycznej. Celem analizy jest sprawdzenie tezy, ze hiperparametry dobierane przez Optune maja ograniczony praktyczny wplyw na jakosc odpowiedzi, a obserwowane roznice wynikow sa zdominowane przez:

- flage `hallucination_flag`,
- opoznienie odpowiedzi (`e2e_ms`, `ttft_ms`),
- roznice miedzy samymi modelami.

Najwazniejszy skrypt to:

```bash
python3 thesis_statistical_analysis.py
```

Skrypt generuje wyniki do katalogu:

```text
thesis-analysis-output/
```

## Wymagane pakiety

Analiza korzysta z typowych pakietow Pythona:

```bash
python3 -m pip install pandas numpy matplotlib seaborn scipy statsmodels tabulate
```

Pakiety zostaly zainstalowane globalnie dla uzytkownika, bez tworzenia venv.

## Struktura danych

Kazdy wiersz JSONL odpowiada jednemu przypadkowi testowemu dla konkretnego modelu, triala i zestawu hiperparametrow.

Najwazniejsze pola:

- `model` - nazwa modelu,
- `trial` - numer triala Optuny,
- `temperature`, `num_ctx`, `num_predict`, `top_p`, `repeat_penalty` - hiperparametry,
- `case_id`, `category` - identyfikator i kategoria przypadku testowego,
- `e2e_ms`, `ttft_ms` - metryki opoznienia,
- `faithfulness`, `answer_relevancy`, `conciseness`, `polish_quality` - oceny czastkowe,
- `hallucination_flag` - flaga halucynacji,
- `composite_score` - wynik laczny.

## Glowna obserwacja matematyczna

W danych zachodzi dokladna zaleznosc:

```text
hallucination_flag = true  => composite_score = 0
hallucination_flag = false => composite_score > 0
```

Dlatego dla sredniego wyniku triala mozna zapisac:

```text
mean_score = (1 - hallucination_rate) * mean_score_given_no_hallucination
```

To oznacza, ze sredni `composite_score` jest w duzej mierze funkcja czestosci flagi halucynacji, a nie subtelnych roznic jakosciowych miedzy odpowiedziami.

Wygenerowany raport pokazuje:

```text
R2(score_mean ~ hallucination_rate) = 0.9800
corr(score_mean, hallucination_rate) = -0.9900
```

Interpretacja: sama czestosc halucynacji wyjasnia okolo 98% zmiennosci sredniego wyniku triali.

## Najwazniejsze artefakty

Raport tekstowy:

```text
thesis-analysis-output/report.md
```

Tabele CSV:

```text
thesis-analysis-output/tables/
```

Najbardziej przydatne tabele:

- `model_summary.csv` - agregaty per dataset i model,
- `trial_summary.csv` - agregaty per trial,
- `score_decomposition.csv` - dekompozycja wyniku przez halucynacje,
- `regression_comparison.csv` - porownanie modeli regresyjnych,
- `incremental_r2.csv` - przyrosty R2 po dodaniu hiperparametrow lub flagi halucynacji,
- `case_summary.csv` - trudnosc poszczegolnych przypadkow testowych,
- `category_summary.csv` - trudnosc kategorii,
- `best_worst_trials.csv` - porownanie najlepszego i najgorszego triala.

Wykresy:

```text
thesis-analysis-output/figures/
```

Kazdy wykres jest zapisany w trzech formatach: `png`, `svg`, `pdf`.

Najbardziej przydatne wykresy do pracy:

- `02_score_vs_hallucination_selected.*` - pokazuje, ze wynik triala spada niemal liniowo z czestoscia halucynacji,
- `03_nonhall_score_spread_selected.*` - pokazuje, ze po odfiltrowaniu halucynacji wyniki sa znacznie stabilniejsze,
- `04_latency_hallucination_pareto_selected.*` - pokazuje kompromis latency vs hallucination rate,
- `05_case_hallucination_rates_selected.*` - pokazuje przypadki testowe najczesciej powodujace halucynacje,
- `07_regression_r2_comparison_selected.*` - porownuje sile wyjasniajaca regresji.

Na wykresach punktowych kazda kropka oznacza jeden trial. Kolor oznacza model. Folder/dataset nie jest kodowany ksztaltem ani kolorem, poniewaz foldery sluza tylko do organizacji plikow z runami.

## Wyniki dla selected runs

Dla zestawu `results-selected-9.05`:

| model | score_mean | hallucination_rate | nonhall_score_mean | e2e_median_ms |
|---|---:|---:|---:|---:|
| gemma3:4b | 0.8837 | 0.0117 | 0.8942 | 31407.7 |
| llama3.2:3b | 0.7321 | 0.1650 | 0.8767 | 27563.4 |
| gemma2:2b | 0.7315 | 0.1833 | 0.8957 | 20866.8 |
| qwen2.5:3b | 0.6967 | 0.2050 | 0.8763 | 30185.8 |

Wniosek:

- `gemma3:4b` ma najwyzszy wynik i bardzo niski hallucination rate, ale jest wolniejszy niz `gemma2:2b`.
- `gemma2:2b` ma dobra jakosc warunkowa po usunieciu halucynacji i najlepsze latency, ale czesciej wpada w flage halucynacji.
- `llama3.2:3b` i `qwen2.5:3b` maja podobna jakosc warunkowa, ale gorszy laczny wynik przez wieksza liczbe halucynacji.

## Regresje i rola hiperparametrow

Dla selected runs i targetu `score_mean`:

| model regresyjny | R2 |
|---|---:|
| hallucination_only | 0.9922 |
| hparams_only | 0.2997 |
| model_only | 0.8201 |
| model_plus_hparams | 0.8587 |
| model_plus_hallucination | 0.9976 |
| full | 0.9982 |

Najwazniejszy wynik metodologiczny:

```text
hparams added after model: delta R2 = 0.0386
hallucination added after model: delta R2 = 0.1775
```

Oznacza to, ze po uwzglednieniu tozsamosci modelu hiperparametry dodaja niewielka czesc wyjasnianej wariancji, natomiast flaga halucynacji dodaje znacznie wiecej.

## Najtrudniejsze przypadki testowe

W selected runs najwieksza czestosc halucynacji wystepuje dla:

| case_id | category | hallucination_rate |
|---|---|---:|
| stale_02 | stale_data | 0.5875 |
| missing_02 | missing_data | 0.5125 |
| missing_01 | missing_data | 0.5000 |
| missing_03 | missing_data | 0.3125 |
| missing_04 | missing_data | 0.2500 |

Wniosek: benchmark najmocniej rozroznia modele na przypadkach `missing_data` i `stale_data`. Przypadki `factual_status` sa w wiekszosci rozwiazane i maja niski hallucination rate.

## Fragment do pracy magisterskiej

Mozliwa formulacja:

> Przeprowadzona analiza wskazuje, ze sredni wynik laczny benchmarku jest silnie zdominowany przez binarna flage halucynacji. W badanym zbiorze kazda odpowiedz oznaczona jako halucynacja otrzymuje `composite_score = 0`, natomiast odpowiedzi bez tej flagi maja wynik dodatni. W konsekwencji sredni wynik triala mozna dokladnie zdekomponowac jako iloczyn odsetka odpowiedzi bez halucynacji oraz sredniej jakosci odpowiedzi niehalucynacyjnych. Regresja liniowa `score_mean ~ hallucination_rate` osiagnela `R2 = 0.9800`, co oznacza, ze sama czestosc halucynacji wyjasnia zdecydowana wiekszosc wariancji srednich wynikow triali.
>
> Po odfiltrowaniu odpowiedzi z flaga halucynacji srednie wyniki warunkowe staja sie znacznie bardziej stabilne miedzy trialami. Dla zestawu `results-selected-9.05` odchylenie standardowe sredniego wyniku warunkowego miedzy trialami jest niewielkie w porownaniu z odchyleniem standardowym lacznego `composite_score`. Sugeruje to, ze hiperparametry dobierane przez Optune nie zmieniaja istotnie semantycznej jakosci odpowiedzi, lecz wplywaja co najwyzej posrednio na czestosc wystapienia flagi halucynacji lub na opoznienie odpowiedzi.
>
> Z praktycznego punktu widzenia dalsze strojenie modeli przy uzyciu tego samego benchmarku ma ograniczona wartosc, jezeli celem jest poprawa jakosci odpowiedzi. Bardziej uzasadnione jest traktowanie wyboru modelu jako problemu Pareto, w ktorym minimalizuje sie opoznienie odpowiedzi przy zachowaniu akceptowalnego poziomu halucynacji.

## Jak odtworzyc analize

1. Upewnij sie, ze w katalogu sa foldery `results-*` z plikami JSONL.
2. Uruchom:

```bash
python3 thesis_statistical_analysis.py
```

3. Otworz raport:

```text
thesis-analysis-output/report.md
```

4. Do pracy wykorzystaj wykresy z:

```text
thesis-analysis-output/figures/
```

Najlepiej uzywac wersji `pdf` albo `svg` przy skladzie pracy, a `png` do szybkiego podgladu.

## Uwagi metodologiczne

Regresje i korelacje sa opisowe, nie przyczynowe. Czesci plikow maja tylko 5 triali, wiec nie nalezy wyciagac silnych wnioskow przyczynowych z pojedynczych korelacji hiperparametrow. Najmocniejszy argument analityczny wynika z dokladnej dekompozycji wyniku oraz ze spadku wariancji po warunkowaniu na `hallucination_flag=false`.
