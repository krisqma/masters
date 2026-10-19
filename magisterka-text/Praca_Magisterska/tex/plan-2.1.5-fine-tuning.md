# Plan: 2.1.5 Fine tuning modeli językowych

Schemat jak w 2.1.2–2.1.4 / Edge–Cloud (uwagi DP z KULMA):  
**otwarcie → opis z literatury → wniosek i kierunek pracy na końcu.**

Przestawić istniejący tekst, nie pisać od zera.

---

## Diagnoza obecnego tekstu

| Akapit | Co jest | Problem DP |
|---|---|---|
| 1 | kontrast kwantyzacja vs fine-tuning | odwołanie „kwantyzacja omówiona **w sekcji powyżej**” — **fałsz**: kwantyzacja jest w **2.2.2 poniżej**; od razu skok bez otwarcia |
| 2 | full FT niepraktyczne → LoRA (Hu2022) | OK literatura; drobne: „nie skupia się…” sklejone ze zdaniem o autorach |
| 3 | QLoRA (Dettmers2023) → „w kontekście niniejszej pracy QLoRA wydaje się niezwykle atrakcyjnym…” | **wniosek w środku** + **mija się z faktami**: w pracy FT **nie** zrobiono; LoRA jest w **planie rozwoju** (`7-podsumowanie`), nie w części badawczej |

Brakuje:

- otwarcia „co będzie w podsekcji”,
- czystego LoRA / QLoRA **bez** „naszej pracy”,
- uczciwego wniosku: w pracy adaptacja **bez** zmiany wag (prompty); FT jako kierunek **przyszły** / poza zakresem empirycznym.

---

## Kolejność akapitów

| # | Co | Treść (z istniejącego tekstu) | Czego nie robić |
|---|---|---|---|
| **0** | Otwarcie (1–2 zd.) | Podsekcja: adaptacja modelu do zadania przez **zmianę wag** (fine-tuning), w tym metody o obniżonym koszcie (PEFT). | Ollama, Wilga, „w naszej pracy”; nie odsyłać do kwantyzacji „powyżej” |
| **1** | Czym jest fine-tuning | Zostaw: FT = modyfikacja wag, adaptacja do zadania. Kontrast z kwantyzacją **pojęciowo** („w odróżnieniu od obniżania precyzji wag przy inferencji…”), **bez** „sekcji powyżej”. Ewentualnie: full FT przy miliardach parametrów lokalnie niepraktyczne (początek obecnego akapitu 2). | Nie mieszać tu LoRA szczegółów |
| **2** | LoRA (Hu2022) | Co to LoRA: zamrożone wagi bazowe + low-rank matrices, redukcja parametrów, merge po treningu, jakość ≈ full FT. | Bez QLoRA; bez wniosku dla pracy |
| **3** | QLoRA (Dettmers2023) | Rozszerzenie LoRA o 4-bit bazę + gradienty w wyższej precyzji; liczby VRAM (7B/24GB, 65B/48GB). **Uciąć** „W kontekście niniejszej pracy QLoRA wydaje się…” | Zero atrakcyjności / wdrożenia w tym akapicie |
| **4** | **Wniosek dla pracy** | „Z przeglądu wynika…”. LoRA/QLoRA umożliwiają dostrajanie SLM przy ograniczonym VRAM. **Kierunek przyjęty w tej pracy:** adaptacja zachowania **bez** fine-tuningu (prompt + kontekst — jak 2.1.3). Fine-tuning (np. LoRA) pozostaje opcją rozwojową / poza zakresem empirycznym niniejszej pracy (spójnie z planem rozwoju). | Nie obiecywać, że „w pracy stosujemy QLoRA” |

---

## Checklista przed oddaniem

- [ ] Brak „kwantyzacja w sekcji powyżej”
- [ ] LoRA i QLoRA opisane **bez** „niniejszej pracy”
- [ ] Wniosek **nie** mówi, że QLoRA jest metodą przyjętą w badaniach
- [ ] Spójność z 2.1.3 (prompty) i z `7-podsumowanie` (LoRA na przyszłość)

---

## Mapowanie: skąd brać zdania

| Docelowy blok | Źródło w obecnym tekście |
|---|---|
| #0 | nowe, 1–2 zdania |
| #1 | akapit 1 (bez „sekcji powyżej”) + początek akapitu 2 o niepraktyczności full FT |
| #2 | reszta akapitu 2 (LoRA) |
| #3 | akapit 3 **bez** ostatniego zdania o „niniejszej pracy” |
| #4 | nowe, w stylu Edge–Cloud; ostatnie zdanie akapitu 3 **przerobić** (nie „atrakcyjny sposób w pracy”, tylko „opisany w literaturze / ewentualnie w rozwoju”) |
