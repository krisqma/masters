# Plan: 2.2.2 Kwantyzacja dla inferencji na sprzęcie bez GPU

Schemat DP jak zwykle: **opis literatury → wniosek na końcu.**

## Zasada nadrzędna: zmiany BARDZO do minimum

Ta podsekcja ma już sensowny ciąg (co to → bandwidth → GPTQ → GGUF/schematy → podsumowanie).  
**Nie przepisywać.** Tylko przestawić / uciąć to, co psuje schemat DP. Zero kosmetyki stylistycznej „przy okazji”.

---

## Diagnoza (tylko to, co trzeba ruszyć)

| Akapit | Co jest | Czy ruszać? |
|---|---|---|
| 1 | definicja + 2 konteksty (FT vs inferencja) + ref do fine-tuningu | **Zostaw.** Już jest otwarciem. |
| 2 | memory-bandwidth, Kim, przykładowe GB/tok/s | **Zostaw.** (RPi w przykładzie liczbowym OK jako ilustracja z lit./rzędu wielkości — nie rozwijać.) |
| 3 | GPTQ (Frantar) → GGUF/llama.cpp → **Ollama w niniejszej pracy** | **Jedyna realna interwencja:** zdanie o Ollama / „niniejszej pracy” **przenieść na koniec** (albo skleić z wnioskiem). W środku ma zostać samo GPTQ + że to podstawa GGUF / llama.cpp. |
| 4 | schematy GGUF (Q4/Q8/Q2…), ARM, MoE | **Zostaw.** |
| 5 | „Warto zaznaczyć…” — miękki wniosek o edge + temperatura | **Zostaw prawie w całości.** Ewentualnie **dopisać 1 zdanie** kierunku: w pracy badana kwantyzacja GGUF (np. Q4_K_M) pod inferencję na urządzeniu o ograniczonych zasobach — **tylko jeśli** nie ma tego już po przeniesieniu zdania o Ollama. |

---

## Minimalna kolejność po poprawce

| # | Akcja | Co dokładnie |
|---|---|---|
| **A** | Nic / prawie nic | Akapity 1, 2, 4 bez zmian treści |
| **B** | Uciąć z akapitu 3 | Zdanie: Ollama = silnik w niniejszej pracy |
| **C** | Wniosek | Albo dokleić to zdanie do akapitu 5, albo zamienić koniec akapitu 5 na krótki „Z przeglądu wynika… / Kierunkiem… GGUF + badanie poziomów kwantyzacji w części badawczej”. **Jedno z dwóch — nie oba na raz, nie rozbudowywać.** |

---

## Czego NIE robić

- Nie dodawać nowego długiego otwarcia „Poniżej omówiono…”
- Nie przenosić LoRA/QLoRA tutaj (to 2.1.5; ref już jest)
- Nie przepisywać GPTQ / GGUF „ładniej”
- Nie ruszać liczb, cytowań, MoE, temperatury — chyba że literówka blokuje zdanie
- Nie mieszać z wynikami z rozdz. 5 (to osobna `subsec:kwantyzacja`)

---

## Checklista (minimum)

- [ ] W akapicie o GPTQ **nie ma** „niniejszej pracy” / Ollama jako wdrożenia
- [ ] Ollama / kierunek pracy **tylko** w ostatnim akapicie (1–2 zdania max)
- [ ] Reszta tekstu = jak była
