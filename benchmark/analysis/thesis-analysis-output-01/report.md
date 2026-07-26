# Analiza statystyczna wyników benchmarków

## Zbiór danych

- Liczba rekordów: 2806
- Liczba podsumowań triali: 142
- Liczba plików: 18
- Liczba modeli: 7

## Główna dekompozycja wyniku

W tym benchmarku każdy rekord z `hallucination_flag=true` otrzymuje `composite_score=0`. Liczba rekordów z halucynacją i niezerowym wynikiem: 0. Liczba rekordów bez halucynacji i zerowym wynikiem: 0.

Dlatego na poziomie triala zachodzi zależność: `mean_score = (1 - hallucination_rate) * mean_score_given_no_hallucination`. Maksymalny błąd numerycznej rekonstrukcji w danych wynosi 0.00000000.

Dla wszystkich triali regresja liniowa `mean_score ~ hallucination_rate` daje R^2=0.9800, korelację Pearsona r=-0.9900 oraz współczynnik kierunkowy -0.9345.

Interpretacja: większość zmienności `composite_score` jest wyjaśniana przez częstość binarnej flagi halucynacji.

## Wybrane runy: podsumowanie modeli

| model       |   rows |   trials |   score_mean |   hallucination_rate |   nonhall_score_mean |   nonhall_score_sd |   e2e_median_ms |   e2e_p90_ms |
|:------------|-------:|---------:|-------------:|---------------------:|---------------------:|-------------------:|----------------:|-------------:|
| gemma3:4b   |    600 |       30 |       0.8837 |               0.0117 |               0.8942 |             0.0943 |      31407.6500 |   41163.5500 |
| llama3.2:3b |    200 |       10 |       0.7321 |               0.1650 |               0.8767 |             0.1326 |      27563.4000 |   34558.6900 |
| gemma2:2b   |    600 |       30 |       0.7315 |               0.1833 |               0.8957 |             0.1146 |      20866.8000 |   27029.7700 |
| qwen2.5:3b  |    200 |       10 |       0.6967 |               0.2050 |               0.8763 |             0.1288 |      30185.8000 |   37527.9500 |

## Wybrane runy: przedziały ufności

| model       |   score_mean |   score_ci_low |   score_ci_high |   hallucination_rate |   hallucination_ci_low |   hallucination_ci_high |   e2e_mean_ms |   e2e_mean_ci_low |   e2e_mean_ci_high |   e2e_median_ms |
|:------------|-------------:|---------------:|----------------:|---------------------:|-----------------------:|------------------------:|--------------:|------------------:|-------------------:|----------------:|
| gemma3:4b   |       0.8837 |         0.8724 |          0.8942 |               0.0117 |                 0.0050 |                  0.0217 |    30401.5477 |        29810.2348 |         31005.7606 |      31407.6500 |
| llama3.2:3b |       0.7321 |         0.6834 |          0.7804 |               0.1650 |                 0.1150 |                  0.2200 |    26800.6610 |        25936.2914 |         27670.0968 |      27563.4000 |
| gemma2:2b   |       0.7315 |         0.7019 |          0.7601 |               0.1833 |                 0.1533 |                  0.2150 |    20528.8797 |        20110.1460 |         20940.9364 |      20866.8000 |
| qwen2.5:3b  |       0.6967 |         0.6453 |          0.7460 |               0.2050 |                 0.1500 |                  0.2600 |    29375.0855 |        28464.0693 |         30287.2723 |      30185.8000 |

## Wybrane runy: dekompozycja według modelu i pliku runu

| model       |   n_trials |   score_sd_between_trials |   hallucination_rate_sd |   nonhall_score_sd_between_trials |   nonhall_sd_to_score_sd_ratio |   r2_score_from_hallucination_rate |
|:------------|-----------:|--------------------------:|------------------------:|----------------------------------:|-------------------------------:|-----------------------------------:|
| gemma2:2b   |         30 |                    0.0494 |                  0.0562 |                            0.0057 |                         0.1149 |                             0.9923 |
| gemma3:4b   |         30 |                    0.0197 |                  0.0215 |                            0.0023 |                         0.1192 |                             0.9865 |
| llama3.2:3b |         10 |                    0.0360 |                  0.0412 |                            0.0057 |                         0.1580 |                             0.9839 |
| qwen2.5:3b  |         10 |                    0.0373 |                  0.0438 |                            0.0094 |                         0.2529 |                             0.9618 |

## Porównanie regresji dla wybranych triali

| regression               |     r2 |   adj_r2 |       aic |
|:-------------------------|-------:|---------:|----------:|
| hallucination_only       | 0.9922 |   0.9921 | -549.7181 |
| hparams_only             | 0.2997 |   0.2421 | -180.1702 |
| model_only               | 0.8201 |   0.8130 | -294.9174 |
| model_plus_hparams       | 0.8587 |   0.8405 | -302.2259 |
| model_plus_hallucination | 0.9976 |   0.9975 | -639.1451 |
| full                     | 0.9982 |   0.9979 | -648.5325 |

## Przyrostowe R² dla wybranych triali

| target             |   model_only_r2 |   hparams_only_r2 | hallucination_only_r2   |   hparams_added_after_model_delta_r2 | hallucination_added_after_model_delta_r2   | hparams_added_after_model_and_hallucination_delta_r2   | hallucination_added_after_model_and_hparams_delta_r2   |
|:-------------------|----------------:|------------------:|:------------------------|-------------------------------------:|:-------------------------------------------|:-------------------------------------------------------|:-------------------------------------------------------|
| e2e_median_ms      |          0.9814 |            0.3871 | 0.3988                  |                               0.0034 | 0.0013                                     | 0.0027                                                 | 0.0005                                                 |
| hallucination_rate |          0.8041 |            0.3049 | NA                      |                               0.041  | NA                                         | NA                                                     | NA                                                     |
| nonhall_score_mean |          0.7014 |            0.1084 | 0.0537                  |                               0.0634 | 0.0109                                     | 0.0623                                                 | 0.0098                                                 |
| score_mean         |          0.8201 |            0.2997 | 0.9922                  |                               0.0386 | 0.1775                                     | 0.0006                                                 | 0.1395                                                 |

Najważniejsze porównanie dotyczy `score_mean`: regresja używająca wyłącznie `hallucination_rate` ma bardzo wysoką siłę wyjaśniającą. Hiperparametry dodają znacznie mniej wyjaśnianej wariancji, gdy uwzględniona jest już tożsamość modelu. Dla `nonhall_score_mean` wartości R² należy interpretować ostrożnie, ponieważ po usunięciu odpowiedzi z halucynacją zakres tej zmiennej jest wąski.

## Porównanie najlepszego i najgorszego triala

| model       |   best_score |   best_hallucination_rate |   best_nonhall_score |   worst_score |   worst_hallucination_rate |   worst_nonhall_score |   score_gap |   hallucination_rate_gap |   nonhall_score_gap |
|:------------|-------------:|--------------------------:|---------------------:|--------------:|---------------------------:|----------------------:|------------:|-------------------------:|--------------------:|
| gemma2:2b   |       0.8473 |                    0.0500 |               0.8919 |        0.6290 |                     0.3000 |                0.8986 |      0.2183 |                   0.2500 |             -0.0067 |
| gemma3:4b   |       0.8966 |                    0.0000 |               0.8966 |        0.8461 |                     0.0500 |                0.8907 |      0.0505 |                   0.0500 |              0.0059 |
| llama3.2:3b |       0.7901 |                    0.1000 |               0.8779 |        0.6962 |                     0.2000 |                0.8702 |      0.0939 |                   0.1000 |              0.0076 |
| qwen2.5:3b  |       0.7491 |                    0.1500 |               0.8812 |        0.6485 |                     0.2500 |                0.8647 |      0.1005 |                   0.1000 |              0.0166 |

## Przypadki testowe najczęściej oznaczane jako halucynacje

| case_id       | category         |   n |   score_mean |   hallucination_rate |   e2e_median_ms |
|:--------------|:-----------------|----:|-------------:|---------------------:|----------------:|
| stale_02      | stale_data       |  80 |       0.3447 |               0.5875 |      22008.1000 |
| missing_02    | missing_data     |  80 |       0.4345 |               0.5125 |      23039.2500 |
| missing_01    | missing_data     |  80 |       0.4648 |               0.5000 |      18517.7500 |
| missing_03    | missing_data     |  80 |       0.6065 |               0.3125 |      27905.3000 |
| missing_04    | missing_data     |  80 |       0.6923 |               0.2500 |      22407.8500 |
| paraphrase_03 | paraphrase       |  80 |       0.8705 |               0.1000 |      24089.5000 |
| conflict_01   | conflicting_data |  80 |       0.6533 |               0.0875 |      36328.2000 |
| factual_01    | factual_status   |  80 |       0.9408 |               0.0125 |      30360.8500 |

## Podsumowanie kategorii

| _dataset              | category         |   n |   score_mean |   hallucination_rate |   e2e_median_ms |   empty_rate |
|:----------------------|:-----------------|----:|-------------:|---------------------:|----------------:|-------------:|
| results-selected-9.05 | missing_data     | 320 |       0.5495 |               0.3937 |      22974.0000 |       0.0000 |
| results-selected-9.05 | stale_data       | 160 |       0.5623 |               0.2938 |      15887.3500 |       0.0000 |
| results-selected-9.05 | paraphrase       | 240 |       0.9287 |               0.0333 |      25891.9000 |       0.0000 |
| results-selected-9.05 | conflicting_data | 240 |       0.6591 |               0.0292 |      34967.0500 |       0.0000 |
| results-selected-9.05 | factual_status   | 640 |       0.9500 |               0.0047 |      25944.0500 |       0.0016 |

## Wykresy

- `figures/01_score_vs_hallucination_all.*`
- `figures/02_score_vs_hallucination_selected.*`
- `figures/03_nonhall_score_spread_selected.*`
- `figures/04_latency_hallucination_pareto_selected.*`
- `figures/05_case_hallucination_rates_selected.*`
- `figures/06_hparam_correlation_heatmap_selected.*`
- `figures/07_regression_r2_comparison_selected.*`

## Wniosek do pracy

Benchmark nie dostarcza silnych dowodów na to, że hiperparametry dobierane przez Optunę istotnie zmieniają semantyczną jakość odpowiedzi. Wynik łączny jest zdominowany przez binarną karę za halucynację. Po usunięciu odpowiedzi z flagą halucynacji wynik warunkowy zmienia się między trialami tylko nieznacznie. W konsekwencji dalsze testowanie modeli na tym samym benchmarku ma ograniczoną wartość jako metoda wnioskowania o hiperparametrach. Bardziej praktycznym kryterium wyboru jest optymalizacja Pareto względem opóźnienia i odsetka halucynacji.

## Zastrzeżenie metodologiczne

Korelacje i modele OLS mają charakter opisowy, a nie przyczynowy. Część plików zawiera tylko pięć triali, więc estymacje wielowymiarowe nie powinny być używane do mocnych twierdzeń przyczynowych. Najsilniejszy argument wynika z dokładnej dekompozycji wyniku oraz z obserwowanego spadku zmienności między trialami po warunkowaniu na odpowiedziach bez flagi halucynacji.
