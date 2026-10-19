# Plan: 2.1.3 Inżynieria promptów i dostarczanie kontekstu

Schemat jak w 2.1.2 / Edge–Cloud (uwagi DP z KULMA):  
**otwarcie → opis wzorców z literatury → wniosek i kierunek pracy na końcu.**

Przestawić istniejący tekst, nie pisać od zera.

---

## Diagnoza obecnego tekstu

| Akapit | Co jest | Problem DP |
|---|---|---|
| 1 | prompt eng + zero/few-shot + CoT + in-context | wszystko w jednym bloku; mini-wniosek o edge/budżecie **w środku** |
| 2 | constraining / system prompt / halucynacje | od razu SLM + asystent — za wcześnie „do pracy” |
| 3 | RAG → od razu „w opisywanym systemie…” | **wniosek w środku opisu** literatury |

Brakuje:

- otwarcia „co będzie w podsekcji”,
- **jednego** akapitu końcowego z kierunkiem pracy.

---

## Kolejność akapitów

| # | Co | Treść (z istniejącego tekstu) | Czego nie robić |
|---|---|---|---|
| **0** | Otwarcie (1–2 zd.) | Podsekcja: jak **kształtować wejście** do modelu i jak **dostarczać kontekst**. Dwa wątki: techniki promptowania + sposoby wstrzykiwania wiedzy zewnętrznej. | Wilga, RPi, prototyp |
| **1** | Inżynieria promptów | Schulhoff + zero/few-shot + CoT. Brown = in-context learning. Wei = CoT. **Uciąć** końcówkę o „ograniczonych zasobach / budżecie” — to wniosek, nie literatura. | Nie mieszać tu RAG-a ani „w naszej pracy” |
| **2** | Constraining / system prompt | Co to jest system prompt; po co constraining; halucynacje przy małych modelach (2–4B) — jako fakt z literatury, nie „wiarygodność asystenta”. Polityka „brak danych → mówię, że nie wiem”. | Bez „w opisywanym systemie”; bez fine-tuningu jako wniosku |
| **3** | RAG (tylko literatura) | Lewis: co to RAG, retrieval, baza wektorowa, doklejanie do kontekstu. **Zatrzymać się tu.** Zdania o deterministycznym wstrzykiwaniu / *structured context injection* → przenieść do #5. | Zero „w tej pracy” w tym akapicie |
| **4** | (opcjonalnie, 1–2 zd.) | Wspólny mianownik: sterowanie zachowaniem **bez zmiany wag** (prompt + kontekst zamiast treningu). | Bez nowego cytatu na siłę |
| **5** | **Wniosek dla pracy** | Jak Edge–Cloud: „Z przeglądu wynika…”. Kierunek: (a) system prompt + reguły antyhalucynacyjne; (b) **nie** klasyczny RAG z retrieval, tylko deterministyczne / strukturyzowane wstrzykiwanie kontekstu z danych IoT (to, co teraz siedzi w środku akapitu RAG). Ewentualnie 1 zdanie, że to czerpie z idei RAG (kontekst zewnętrzny), ale bez wyszukiwania wektorowego. | Tego nie wcześniej |

---

## Checklista przed oddaniem

- [ ] RAG opisany **bez** Twojego systemu
- [ ] „W opisywanym systemie / structured injection” **tylko** w ostatnim akapicie
- [ ] Edge / budżet / asystent **nie** w środku opisu technik
- [ ] Jeden kierunek na końcu: prompt + kontekst IoT, nie fine-tuning

---

## Mapowanie: skąd brać zdania

| Docelowy blok | Źródło w obecnym tekście |
|---|---|
| #0 | nowe, 1–2 zdania (jak otwarcie 2.1.2) |
| #1 | akapit 1, bez końcówki o zasobach/budżecie |
| #2 | akapit 2, bez „wiarygodność asystenta” / tonu wdrożeniowego |
| #3 | początek akapitu 3 (Lewis + klasyczny RAG), **bez** zdań o „opisywanym systemie” |
| #5 | końcówka akapitu 3 (deterministyczne wstrzykiwanie, structured context injection) + ewentualnie końcówka akapitu 1 (zasoby / bez treningu) |
