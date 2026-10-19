# Plan: 2.1.4 Optymalizacja hiperparametrów generacji

Schemat jak w 2.1.2 / 2.1.3 / Edge–Cloud (uwagi DP z KULMA):  
**otwarcie → opis z literatury → wniosek i kierunek pracy na końcu.**

Przestawić istniejący tekst, nie pisać od zera.

---

## Diagnoza obecnego tekstu

| Akapit | Co jest | Problem DP |
|---|---|---|
| 1 | lista hiperparametrów + że optymalne zależą od zadania/sprzętu + grid search za drogi | OK merytorycznie, ale **brak otwarcia**; od razu lista „takie jak:” (DP nie lubi `:` w środku zdania) |
| 2 | Optuna / TPE (Akiba2019) → od razu „znajduje zastosowanie w SLM + Ollama” | **wniosek wdrożeniowy w środku** literatury; Ollama to już Wasze narzędzie, nie przegląd |

Brakuje:

- otwarcia „co będzie w podsekcji”,
- czystego opisu metody **bez** Ollamy / Waszego SLM,
- **jednego** akapitu końcowego z kierunkiem pracy (Optuna w części badawczej).

Uwaga: podsekcja jest krótka (2 akapity) — nie rozdmuchiwać. Wystarczy 0 → 1 → 2 → 3.

---

## Kolejność akapitów

| # | Co | Treść (z istniejącego tekstu) | Czego nie robić |
|---|---|---|---|
| **0** | Otwarcie (1–2 zd.) | Podsekcja: jak dobierać **parametry generacji** (temperatura, top-p, długość itd.), gdy pełne przeszukanie siatki jest nieopłacalne. | Wilga, RPi, Ollama, „w naszej pracy” |
| **1** | Problem / przestrzeń parametrów | Zostaw listę parametrów (temperatura, context window, max długość, top-p, kara za powtórzenia) + że optymalne wartości zależą od zadania i sprzętu + że grid search jest kosztowny. Ewentualnie rozbić `takie jak:` na zdania / krótką listę (uwaga DP o `:`). | Nie wrzucać tu Optuny ani Ollamy |
| **2** | Optuna / TPE (tylko literatura) | Akiba2019: co to Optuna, bayesowskie próbkowanie, TPE, model funkcji celu, mniej ewaluacji niż grid/losowe. **Zatrzymać się na „skuteczna w optymalizacji wieloparametrowej”.** Zdanie o SLM + Ollama → przenieść do #3. | Zero „aplikacji Ollama” w tym akapicie |
| **3** | **Wniosek dla pracy** | „Z przeglądu wynika…”. Kierunek: w części badawczej dobór hiperparametrów generacji SLM przez Optunę (TPE), zamiast pełnej siatki — bo przestrzeń parametrów × sprzęt edge jest za duża na ręczne / grid. Tu wolno: Ollama, SLM, urządzenie o ograniczonych zasobach. | Tego nie wcześniej |

---

## Checklista przed oddaniem

- [ ] Optuna opisana **bez** Ollamy i bez „naszego” SLM
- [ ] Ollama / kierunek badawczy **tylko** w ostatnim akapicie
- [ ] Brak skoku „metoda → od razu nasz stack”
- [ ] Jeden kierunek na końcu: Optuna (TPE) do doboru parametrów generacji

---

## Mapowanie: skąd brać zdania

| Docelowy blok | Źródło w obecnym tekście |
|---|---|
| #0 | nowe, 1–2 zdania |
| #1 | cały obecny akapit 1 (ew. bez `:` / drobna redakcja listy) |
| #2 | akapit 2 **bez** ostatniego zdania o SLM + Ollama |
| #3 | ostatnie zdanie akapitu 2 („Znajduje swoje zastosowanie… Ollama”) rozwinięte w stylu „Z przeprowadzonego przeglądu wynika… Kierunkiem przyjętym…” |
