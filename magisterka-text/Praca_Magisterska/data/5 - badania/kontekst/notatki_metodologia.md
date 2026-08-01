# Notatki do rozmowy o benchmarku — 2026-05-13

## 1. Co dokładnie benchmark testuje

Benchmark nie odpala pełnego produkcyjnego pipeline'u `context_engine -> Teacher -> Student`.

Zamiast tego testuje modele na kontrolowanych przypadkach złożonych z:

- `system_prompt` — tekstowy opis stanu domu zbudowany offline z jednego wiersza CSV z sensorami
- `question` — pytanie użytkownika do tego kontekstu
- `golden_answer` — wzorcowa odpowiedź referencyjna
- `model_answer` — odpowiedź wygenerowana przez badany model podczas benchmarku

Ważne: `golden_answer` nie jest używane do prostego porównania `tekst == tekst`. Ewaluator LLM dostaje jednocześnie `system_prompt`, `question`, `golden_answer` i `model_answer`, a potem wystawia oceny jakościowe.

Najkrótszy schemat:

`CSV -> system_prompt -> question -> model_answer -> LLM judge porównuje do kontekstu i golden_answer`

## 2. Skąd biorą się dane

Źródłem jest plik:

- `db_sampler/sensor_data_2025-10-31_2026-01-27.csv`

Generator benchmarku:

1. czyta realne dane sensorowe z CSV,
2. wybiera określone wiersze,
3. zamienia każdy wybrany wiersz na tekstowy `system_prompt`,
4. generuje przez OpenRouter `question` i `golden_answer`,
5. zapisuje gotowe przypadki do `golden.jsonl`.

To znaczy:

- dane źródłowe są prawdziwe, pochodzą z CSV,
- benchmark jest offline i deterministycznie buduje zestaw przypadków,
- OpenRouter służy do przygotowania benchmarku i do późniejszej ewaluacji odpowiedzi.

## 3. Czy to idzie przez OpenRouter

Tak, ale są dwa osobne miejsca:

### Generowanie benchmarku

Podczas tworzenia `golden.jsonl` OpenRouter generuje:

- `question`
- `golden_answer`

W kodzie używany jest tu model:

- `anthropic/claude-sonnet-4-6`

### Ewaluacja odpowiedzi modeli

Po uruchomieniu benchmarku odpowiedź testowanego modelu jest oceniana przez osobny model sędziujący przez OpenRouter.

Ten model nie jest zahardkodowany jako konkretny Haiku w kodzie, tylko jest brany ze zmiennej środowiskowej:

- `HAIKU_MODEL`

Czyli poprawne zdanie na rozmowę brzmi:

„Tak, ocena odpowiedzi leci przez OpenRouter do modelu ewaluacyjnego skonfigurowanego w `HAIKU_MODEL`, a wygenerowanie pytań i goldenu było robione przez `Claude Sonnet`.”

## 4. Co oznaczają metryki jakości

Ewaluator zwraca pięć pól:

- `faithfulness`
- `answer_relevancy`
- `conciseness`
- `polish_quality`
- `hallucination_flag`

### `faithfulness`

To najważniejsza metryka.

Pytanie brzmi:

„Czy każde twierdzenie faktyczne w odpowiedzi modelu wynika z dostarczonego kontekstu?”

Interpretacja:

- `1.0` — odpowiedź jest w pełni zgodna z kontekstem
- `0.0` — model podaje fakty, których nie ma w kontekście

To w praktyce mierzy, czy model nie zmyśla i czy nie wykracza poza dane sensorowe.

### `answer_relevancy`

Pytanie brzmi:

„Czy odpowiedź rzeczywiście odpowiada na zadane pytanie?”

Interpretacja:

- wysoki wynik — odpowiedź trafia dokładnie w pytanie
- niski wynik — model mówi obok tematu, odpowiada za szeroko albo nie podaje tego, o co pytano

Model może być wierny kontekstowi, ale mało relewantny, jeśli zamiast podać konkretną temperaturę opisze ogólny stan domu.

### `conciseness`

Pytanie brzmi:

„Czy odpowiedź jest krótka i bez zbędnych informacji?”

Interpretacja:

- wysoki wynik — odpowiedź jest zwięzła
- niski wynik — odpowiedź jest rozwlekła, zawiera disclaimery, dygresje albo niepotrzebne dodatki

### `polish_quality`

To ocena naturalności i poprawności języka polskiego:

- gramatyka
- naturalne sformułowania
- brak dziwnej składni
- brak anglicyzmów

Ta metryka jest liczona i zapisywana, ale nie wchodzi do aktualnego `composite score`.

### `hallucination_flag`

To flaga binarna:

- `true` — odpowiedź zawiera konkretne fakty spoza kontekstu
- `false` — odpowiedź nie halucynuje

Jeśli `hallucination_flag = true`, to cały `composite_score` jest zerowany do `0.0`.

## 5. Dokładny wzór na composite score

Aktualny wzór składa się z:

- `faithfulness` z wagą `0.40`
- `answer_relevancy` z wagą `0.30`
- `conciseness` z wagą `0.20`
- opóźnienia z łączną wagą `0.10`

Opóźnienie zostało rozbite na dwa składniki:

- `E2E` z wagą `0.05`
- `TTFT` z wagą `0.05`

Dokładnie:

```text
if hallucination_flag:
    score = 0.0
else:
    norm_e2e  = min(e2e_ms / 90000, 1.0)
    norm_ttft = 1.0 if ttft_ms is None else min(ttft_ms / 45000, 1.0)
    latency_bonus = 0.05 * (1 - norm_e2e) + 0.05 * (1 - norm_ttft)

    score =
      0.40 * faithfulness +
      0.30 * answer_relevancy +
      0.20 * conciseness +
      latency_bonus
```

Ważne:

- `polish_quality` nie wchodzi do tego wzoru
- halucynacja zeruje wynik niezależnie od jakości językowej i relewancji

## 6. Co oznaczają kategorie przypadków

W benchmarku jest 20 przypadków:

- `8` `factual_status`
- `4` `missing_data`
- `3` `conflicting_data`
- `2` `stale_data`
- `3` `paraphrase`

### `factual_status`

To zwykłe pytania o stan domu na podstawie poprawnego, bogatego kontekstu.

Przypadki biorą się z takich wierszy CSV, w których jest co najmniej 8 dostępnych odczytów sensorów.

Typowe pytania:

- jaka jest temperatura w salonie
- jaka jest wilgotność w łazience
- czy okno jest otwarte
- jaka jest moc bojlera

To jest podstawowy test: czy model umie poprawnie odczytać fakty z kontekstu.

### `missing_data`

Tak, to dokładnie są pytania o dane, których w danym `system_prompt` nie ma.

Mechanizm:

- generator wybiera rzadsze wiersze z tylko częściowo dostępnymi danymi,
- potem model generujący benchmark ma ułożyć pytanie o sensor, którego nie ma w promptcie,
- `golden_answer` ma wtedy powiedzieć wprost, że brak danych.

To testuje, czy model potrafi:

- przyznać brak informacji,
- nie zgadywać,
- nie halucynować brakującego odczytu.

### `conflicting_data`

To nie są konflikty sztucznie wymyślone ręcznie, tylko wybierane z realnego CSV.

Generator szuka wierszy, gdzie w salonie jednocześnie występują dwa odczyty temperatury:

- z czujnika powietrza
- z czujnika klimatu

i różnią się o więcej niż `1.0` stopnia.

Czyli konflikt pochodzi bezpośrednio z danych.

Dopiero później benchmark narzuca interpretację zadania:

- pytanie ma dotyczyć temperatury salonu,
- wzorcowa odpowiedź ma wskazać, że kontekst zawiera sprzeczne dane.

To testuje, czy model:

- zauważy konflikt,
- nie wybierze arbitralnie jednej wartości,
- nie udaje pewności tam, gdzie dane są niespójne.

### `stale_data`

Tu dane też są realne, ale benchmark jawnie zaznacza, że są stare.

Mechanizm:

- wybierane są wcześniejsze wiersze z CSV,
- do `system_prompt` dopisywana jest informacja, że ostatni odczyt pochodzi sprzed określonej liczby godzin,
- `golden_answer` ma podkreślić nieaktualność danych.

To testuje, czy model rozumie problem świeżości danych, a nie tylko samą treść odczytu.

### `paraphrase`

To są te same konteksty co w pierwszych trzech przypadkach `factual_status`, ale pytanie jest przeformułowane inaczej.

Czyli:

- ten sam stan domu,
- to samo znaczenie pytania,
- inne sformułowanie językowe.

To testuje odporność modelu na parafrazy i różne sposoby zadania tego samego pytania.

## 7. Jak o tym mówić jednym zdaniem

Najprostsza poprawna wersja:

„Benchmark bierze realne dane z CSV sensorów, buduje z nich tekstowy kontekst, do tego generuje pytanie i wzorcową odpowiedź, a potem sprawdza, jak lokalny model odpowiada i jak jego odpowiedź ocenia LLM-judge przez OpenRouter pod kątem zgodności z kontekstem, trafności i zwięzłości.”

## 8. Dwie ważne pułapki na rozmowę

### To nie jest zwykłe porównanie odpowiedzi do wzorca

Nie mów:

„Porównujemy odpowiedź modelu z `golden_answer`.”

Lepiej mów:

„`golden_answer` jest odpowiedzią referencyjną, a końcowej oceny dokonuje LLM-judge na podstawie całego kontekstu, pytania, wzorca i odpowiedzi modelu.”

### Benchmark nie używa pełnego `Context Engine`

Nie mów:

„Benchmark testuje dokładnie ten sam pipeline co aplikacja.”

Lepiej mów:

„Benchmark upraszcza pipeline aplikacyjny i testuje modele na kontrolowanych parach `kontekst + pytanie`, zbudowanych offline z rzeczywistych danych sensorowych.”
