# Wilga: wyniki naprawy stabilności (20-21.10.2026)

Diagnoza: [RAPORT.md](RAPORT.md). Przyczyna: gemma3 ma okno przesuwne 1024 tokenów. Gdy rozmowa w jedynym slocie KV Ollamy je przekroczy, każda zmiana promptu oznacza przeliczenie całości (~7 tok/s na RPi), czyli timeouty i `500 | 2m0s`.

Naprawa: **budżet tokenów**, który trzyma każde żądanie do Ucznia w oknie, plus usunięcie zbędnych rozbieżności promptu i krótszy prompt sprawdzony testem jakości.

## Przed i po (te same narzędzia i pytania z prawdziwych rozmów)

| | Przed naprawą | Po naprawie (finalny kod, prawdziwy `.env`) |
|---|---|---|
| Rozmowa z 30.07 (replay, 21 tur) | timeouty od 4.-5. tury, potem `500 \| 2m0s` | **21/21 OK**, zwykłe tury p95 20,6 s |
| Wszystkie tury (V1-V3 + 30 min soak) | od pewnego momentu ~100% timeoutów | **60/60 OK** (+1 celowy abort), **0 timeoutów, 0 × 5xx/503** |
| Czas do pierwszego słowa odpowiedzi | 13-84 s, potem nigdy | **p50 4,2 s, p95 17,7 s** |
| Cała tura E2E | 6-24 s, dopóki działało | **p50 8,4 s**; p95 35 s, ale to wyłącznie zamierzone przypadki (patrz niżej) |
| Tura z przycięciem historii | (okno przesuwało się i psuło cache) | **p50 19,5 s, max 28 s** (9 przypadków) |
| Rozmiar slotu na RPi | 1030-1300 tokenów (> okno) | **max 904** (limit 1024) |
| Rozgrzewanie sesji | 60-131 s, efekt i tak tracony | **≤ 31 s**, pierwsze pytanie dopisuje się do cache |
| Pytanie w trakcie rozgrzewania | 503 „Model się rozgrzewa” | czeka na model (22 s) i dostaje odpowiedź |
| Przeładowania modelu | możliwe przy różnych `options` | **0** (journal RPi) |
| Restart RPi (odłączenie zasilania) | ręczne odkręcanie | watchdog sam wykrył brak modelu i rozgrzał go ponownie |
| Historia | sztywne 6 wiadomości, obcinane `.strip()` | dynamiczna, surowa, w backendzie; zwykle 4-8 wiadomości, w małych sesjach do 14 |

Zwykłe tury powyżej 25 s w finalnym przebiegu to dokładnie 3 przypadki:
- pytanie celowo zadane w trakcie rozgrzewania (V2);
- celowo wymuszona najdłuższa odpowiedź (V3, 128 tokenów, 58 s);
- jedna odpowiedź o maksymalnej długości w soaku (38 s).

Odpowiedź do 128 tokenów przy ~4 tok/s to do ~32 s samej generacji, ale frontend czyta ją zdanie po zdaniu od pierwszego chunka.

Runy (logi w `runs/`, poza gitem):
- `20261021T0054Z-final2` (finalny kod);
- `20261020T2045Z-final`, `20261020T1938Z-fixed`, `20261020T2000Z-fixed-soak` (wcześniejsze iteracje).

Liczby z `python acceptance.py <run>`.

## Jakość odpowiedzi: A/B starego i nowego promptu

Ten sam model na RPi, te same opcje, 5 wierszy CSV × 8 pytań z prawdziwych rozmów. Złote odpowiedzi daje Teacher, oceny sędzia LLM (`benchmark/evaluate.py`), a deterministycznie sprawdzane są liczby. Pełna tabela: [quality_ab_summary.md](quality_ab_summary.md).

Iteracje:
- **new:** pierwsza kompresja;
- **new2:** jawne „brak …” w faktach;
- **new3:** dobitna reguła o liczbach;
- **new4 (finalny):** new3 plus zdanie „cytuj tylko, gdy pytanie dotyczy stanu domu” tuż przy danych.

| | stary prompt | **finalny (new4)** |
|---|---|---|
| Wierność kontekstowi | 0,455 | **0,568** |
| Średnia 4 ocen sędziego | 0,669 | **0,686** |
| Oznaczone jako halucynacja | 72,5% | **57,5%** |
| Odpowiedzi z liczbą spoza danych | 7,5% (3/40) | **2,5% (1/40)** |
| Persona/off-topic bez liczb | 10/10 | **10/10** |
| Średnia długość odpowiedzi | 61 tok | **47 tok** |

Wnioski z iteracji:
- Pierwsza kompresja (new) skracała odpowiedzi, ale częściej zmyślała liczby przy brakujących odczytach (25%) i dorzucała odczyty do pytań o personę.
- Pomogły jawne braki w faktach plus dobitna reguła o liczbach.
- Kluczowe było też zdanie o użyciu danych **bezpośrednio przy danych**; reguła tylko w prefiksie nie wystarczała.

Pozostały 1 przypadek: łazienka bez odczytu temperatury, a model podał 22.3°C. Stary prompt w tych samych sytuacjach podawał temperatury urządzeń z innych pokoi albo wymyślał pomieszczenia („Salon”, „Sypialnia”).

## Co zmieniono

1. **Prompt Ucznia** (`context_service/app/context_engine.py`, `data_source.py`, `teacher_client.py`):
   - prefiks statyczny 596 → ~390 tokenów, z dobitną regułą „liczbę podajesz TYLKO, gdy stoi w bloku”;
   - fakty: jedna linia na pokój, jawne `brak temp./wilg./ciśn.` lub `brak odczytów`, liczniki w kWh, bez temperatur wewnętrznych urządzeń; linia „cytuj tylko, gdy pytanie dotyczy stanu domu” przy danych;
   - Findings maks. 160 znaków.
2. **Budżet tokenów** (`whisper-proj/backend/app/student_prompt.py`):
   - `prompt + num_predict + 32 ≤ 1024`;
   - prefiks i tury liczone dokładnie z liczników Ollamy, pytanie estymowane (błąd ≤ 4%);
   - historia dynamiczna z histerezą; po przycięciu zostaje ≤ 120 tokenów, więc przycięcie kosztuje ok. 15-20 s co kilka tur;
   - awaryjnie (największe wiersze danych) zrzucane są kolejno: Findings, liczniki energii, „Instalacja ogólna”, „Na zewnątrz”.
3. **Historia w backendzie** (`session_state.py`):
   - surowe odpowiedzi;
   - tylko pełne wymiany;
   - `client_session_id` resetuje historię po przeładowaniu strony lub „Nowej rozmowie”;
   - zasiew z frontendu tylko po restarcie backendu.
4. **Spójne żądania:**
   - te same opcje z pracy (`temperature 0.2`, `num_ctx 2048`, `top_p 0.9`, `repeat_penalty 1.05`) i `keep_alive: -1` we wszystkich żądaniach;
   - zamiast pingów co 10 min watchdog `/api/ps` co 60 s.
5. **Timeouty i aborty:**
   - frontend: 200 s do pierwszego bajtu, 30 s między chunkami; backend: 180 s;
   - zerwanie połączenia przez przeglądarkę od razu zatrzymuje RPi;
   - czat czeka na model zamiast zwracać 503.
6. **Logi:** backend zawsze loguje INFO; każda tura to linia `Chat done: ... prompt_tok slot_tok prefill_s hist_turns trimmed ...`.

## Ograniczenia

- **Termika RPi.** Po kilku godzinach ciągłego ciężkiego obciążenia (test A/B ze starym promptem) RPi dochodziło do 85-88°C i chwilami do miękkiego limitu 80°C. Na RPi nie ma wentylatora widocznego dla systemu (`cooling_fan`). W normalnym użyciu temperatury mieściły się w 45-70°C. Przy długiej ciągłej pracy warto dodać aktywne chłodzenie.
- **Mniej historii przy dużych odczytach.** Gdy dane są bardzo duże (wiele urządzeń z odczytami), na historię zostaje 1-2 tury. Mediana w zbiorze to ~2-3 tury (4-6 wiadomości).
- **Ręczne żądania do RPi.** Każde żądanie z innymi `options` (np. `curl` bez `num_ctx 2048`) przeładuje model; jest to opisane w README.
- **Prefill RPi jest ograniczony mocą CPU.** `num_batch` 32/128/512 nie ma znaczenia, a `num_thread=3` jest wolniejsze (Faza 0, `runs/phase0`).
