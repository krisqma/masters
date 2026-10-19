# Plan: 2.3.1 Interfejsy głosowe (VAD / STT / TTS)

Ostatnia podsekcja z tabelki „nieredagowanych”.  
Schemat DP: **otwarcie → VAD → STT → TTS (sama literatura) → wniosek na końcu.**

Przestawić istniejący tekst; nie pisać od zera. Zmiany trzymać **umiarkowanie małe** (więcej niż w 2.2.2, bo tu wnioski siedzą w środku, ale bez kosmetycznego przepisywania całości).

---

## Diagnoza obecnego tekstu

| Akapit | Co jest | Problem DP |
|---|---|---|
| — | brak otwarcia pod `\subsubsection` | „pod paragrafem nie może być pusto” / nie wiadomo, co będzie w podsekcji |
| 1 | VAD → Silero | OK literatura; na końcu „idealne rozwiązanie dla IoT” = mini-wniosek za wcześnie; literówki (`latency`, podwójna kropka) |
| 2 | Whisper / faster-whisper | **Wilga w środku** („dom wczasowy… ośrodek w Wildze”) — klasyczne „po co mieszać Wilgę”; „skrojony idealnie pod opisywane rozwiązanie” = wniosek w środku |
| 3 | TTS: ElevenLabs, VITS, edge-tts | urwane — **brak wniosku dla pracy**; edge-tts wymienione, ale bez kierunku (a w rozdz. 6 jest ZofiaNeural + świadomość chmury) |

Fakt z realizacji (żeby wniosek był spójny, nie kłamał):

- VAD w prototypie = **prosty detektor poziomu** w przeglądarce, **nie** Silero (Silero tylko w przeglądzie),
- STT = **faster-whisper** (`small`, PL),
- TTS = **edge-tts** (`pl-PL-ZofiaNeural`).

---

## Kolejność akapitów

| # | Co | Treść (z istniejącego tekstu) | Czego nie robić |
|---|---|---|---|
| **0** | Otwarcie (1–2 zd.) | Łańcuch interakcji głosowej: detekcja mowy (VAD) → rozpoznawanie (STT) → synteza (TTS). Poniżej krótki przegląd rozwiązań. | Wilga, Wilguś, RPi |
| **1** | VAD | Zostaw: po co VAD, start-of-speech, Silero (rozmiar, latency, MIT, 8/16 kHz). **Uciąć** „idealne rozwiązanie dla IoT” (albo przenieść sens do #4 jako opcję z literatury). | Bez „nasz system”; bez wyboru energii vs Silero tutaj |
| **2** | STT | Whisper: weak supervision, 680k h, PL, warianty rozmiaru, faster-whisper pod ograniczone zasoby. **Uciąć / przenieść:** Wilga + „skrojony idealnie pod opisywane rozwiązanie”. Zostaw fakt o odporności na szum jako właściwość modelu (bez ośrodka). | Zero Wilgi w środku |
| **3** | TTS | Zostaw: rynek szeroki, ElevenLabs (komercyjne), VITS (open, end-to-end), edge-tts + głos PL ZofiaNeural. **Bez** „wybraliśmy Zofię / Wilguś” — to do #4. | Bez Aktu AI / głosu dziecięcego tutaj (to realizacja) |
| **4** | **Wniosek dla pracy** | „Z przeglądu wynika…”. Kierunek: pełny łańcuch VAD→STT→TTS. Konkretnie: STT w oparciu o Whisper/faster-whisper; TTS przez edge-tts (PL). VAD: w literaturze Silero jako lekki model brzegowy; w prototypie przyjęto prostszy detektor poziomu po stronie klienta (lżejszy niż neuronowy VAD) — spójnie z rozdz. 6. Ewentualnie 1 zd. że edge-tts jest zależny od sieci → kierunek rozwoju lokalnego TTS (jak w realizacji). | Nie w środku opisu Whisper/Silero |

---

## Checklista

- [ ] Jest krótkie otwarcie (VAD → STT → TTS)
- [ ] Brak Wilgi / „opisywanego rozwiązania” w środku STT
- [ ] Silero = literatura; wybór prostego VAD dopiero we wniosku (albo tylko w rozdz. 6 — wtedy we wniosku nie obiecywać Silero w prototypie)
- [ ] edge-tts / Whisper jako kierunek **na końcu**
- [ ] Jedna końcówka „Z przeglądu wynika…”, nie trzy mini-reklamy w akapitach

---

## Mapowanie: skąd brać zdania

| Docelowy blok | Źródło |
|---|---|
| #0 | nowe, 1–2 zdania |
| #1 | akapit 1 bez końcówki „idealne dla IoT” |
| #2 | akapit 2 bez Wilgi i bez „skrojony idealnie…” |
| #3 | akapit 3 bez zmian merytorycznych |
| #4 | nowe + wyrzucone końcówki z #1/#2 + spójność z `6-realizacja` (energia VAD, faster-whisper, edge-tts) |

---

## Uwaga o zakresie zmian

Więcej przestawień niż w kwantyzacji (tam wystarczyło jedno zdanie). Tu **minimum strukturalne** = otwarcie + wyciecie Wilgi/idealności ze środka + jeden wniosek.  
Nie poprawiać całego stylu / wszystkich literówek „przy okazji”, chyba że blokują zdanie (np. podwójna kropka przy Silero).
