# Analiza Wideo z Monitoringu — Log Decyzji Projektowych

> Dokument dla zleceniodawcy. Zawiera wszystkie rozpatrywane opcje, ich zalety/wady, podjete decyzje i uzasadnienia, oraz pelny flow klienta i inwestora.
>
> **Status 2026-07-17:** to jest żywy log decyzji. Pierwotne wybory pozostają
> jako historia, a sekcje „Aktualizacja” i tabela podsumowania opisują aktualny
> system. Bieżący kontrakt techniczny znajduje się w `SPEC.md`.

---

## 1. Spis tresci

1. [Opis koncepcji](#2-opis-koncepcji)
2. [Decyzja 1: Frigate vs. custom pipeline](#3-decyzja-1-silnik-analizy-wideo)
3. [Decyzja 2: Glebokosc raportu (poziom analizy)](#4-decyzja-2-glebokosc-raportu)
4. [Decyzja 3: Dostarczanie wideo od klienta](#5-decyzja-3-dostarczanie-wideo-od-klienta)
5. [Decyzja 4: Tracking osob](#6-decyzja-4-tracking-osob)
6. [Decyzja 5: Prototyp vs. integracja z platforma](#7-decyzja-5-prototyp-vs-integracja)
7. [Decyzja 6: Stack technologiczny](#8-decyzja-6-stack-technologiczny)
8. [Decyzja 7: Format raportu](#9-decyzja-7-format-raportu)
9. [Decyzja 8: RODO i kwestie prawne](#10-decyzja-8-rodo-i-kwestie-prawne)
10. [Flow klienta (pelny scenariusz)](#11-flow-klienta)
11. [Flow inwestora (pelny scenariusz)](#12-flow-inwestora)
12. [Podsumowanie decyzji](#13-podsumowanie-decyzji)
13. [Ryzyka i ograniczenia](#14-ryzyka-i-ograniczenia)

---

## 2. Opis koncepcji

Wykorzystanie mocy obliczeniowej GPU na farmach miningowych (posiadanych przez inwestorow) do analizy wideo z systemow monitoringu klientow.

**Cel:** Klient udostepnia obraz z kamery monitoringu → system analizuje wideo na GPU → generuje raport aktywnosci osob (np. "osoba siedziala 15min, chodzila 30min, stala 10min").

**Kontekst infrastrukturalny:**
- 10 serwerow GPU, ~60x RTX 5070 (12GB VRAM), rozproszonych po Polsce
- Serwery za NATem, brak statycznych IP, polaczenia outbound-only
- Dual-boot GRUB: HiveOS (mining) / Ubuntu (AI) — przelaczenie ~3-5 min
- Istniejaca architektura job dispatch (pull-based, presigned URLs, R2 storage)

---

## 3. Decyzja 1: Silnik analizy wideo

### Opcja A: Frigate NVR (https://frigate.video)

Gotowy system NVR z wbudowana detekcja obiektow AI.

| | Opis |
|---|---|
| **Zalety** | Gotowy tracking osob, strefy (zones), atrybuty (twarz, poza), dojrzaly ekosystem, duza spolecznosc |
| **Wady** | Zaprojektowany do **live RTSP streaming**, nie do batch analizy plikow MP4. Brak wbudowanego job queue. Zeby przetworzyc plik wideo trzeba emitowac go jako fake RTSP stream (ffmpeg → rtsp-simple-server) — nieefektywne i kruche. Wymaga MQTT brokera. Konfiguracja per-"kamera" dla kazdego joba. Ciezka integracja z naszym job systemem |
| **Konsekwencje** | Duzy narzut integracyjny, architektura niekompatybilna z batch processing, trudna automatyzacja |

### Opcja B: Custom pipeline (YOLO-pose + heurystyki) ✅ WYBRANA

Wlasny pipeline zbudowany na istniejacym YOLO + ffmpeg.

| | Opis |
|---|---|
| **Zalety** | Pelna kontrola nad pipeline. Wpasowuje sie w istniejacy job flow (pull-based, R2, Docker). Lzejsze i prostsze na MVP. Reuse istniejacego kodu YOLO (yolo-serve). YOLO-pose daje 17 keypoints per osoba — wystarczajace do klasyfikacji aktywnosci |
| **Wady** | Trzeba dopisac: ekstrakcje klatek, tracking (jesli potrzebny), klasyfikacje aktywnosci, generowanie raportu |
| **Konsekwencje** | Wiecej pracy wlasnej, ale pelna kompatybilnosc z platforma. Pipeline latwiej rozszerzalny o nowe typy analizy |

### Uzasadnienie

Frigate jest doskonalym narzedziem do **ciaglego monitoringu live kamer** (np. smart home), ale **nie jest zaprojektowane do batch przetwarzania plikow wideo**. Nasz use case to: klient daje nagranie → system przetwarza → zwraca raport. To batch job, nie live monitoring. Custom pipeline na YOLO-pose idealnie wpasowuje sie w istniejaca architekture (Docker + GPU + pull-based jobs + R2).

---

## 4. Decyzja 2: Glebokosc raportu

### Poziom 1: Detekcja + zliczanie

- "Wykryto 3 osoby, laczna obecnosc 45min, heatmapa stref aktywnosci"
- Bez rozrozniania siedzi/stoi/chodzi

| | Opis |
|---|---|
| **Zalety** | Najprostsze do zbudowania. Wystarczy YOLO standard (bez pose). Wysoka dokladnosc detekcji |
| **Wady** | Malo wartosciowy raport. Nie odpowiada na pytanie "czym zajmowali sie ludzie" |

### Poziom 2: Zone tracking

- "Osoba w strefie A: 20min, osoba w strefie B: 25min"
- Wymaga trackingu + definicji stref

| | Opis |
|---|---|
| **Zalety** | Bardziej wartosciowy niz poziom 1. Pokazuje gdzie ludzie spedzaja czas |
| **Wady** | Wymaga per-person trackingu (ByteTrack/DeepSORT). Klient musi definiowac strefy — dodatkowa zlonosc UI |

### Poziom 3: Klasyfikacja aktywnosci ✅ WYBRANY

- "Lacznie: 45 osobo-minut stania, 20 osobo-minut siedzenia, 15 osobo-minut chodzenia"
- YOLO-pose (17 keypoints) + heurystyki geometryczne

| | Opis |
|---|---|
| **Zalety** | Najwartosciowszy raport — odpowiada wprost na pytanie zleceniodawcy. Nie wymaga definicji stref. 4 aktywnosci: siedzi/stoi/chodzi/biega. Rule-based (bez dodatkowego modelu ML) |
| **Wady** | Wymaga YOLO-pose (nie standard YOLO). Heurystyki moga byc niedokladne przy trudnych katach kamery. Target dokladnosci: 80% |
| **Konsekwencje** | Uzycie `yolo11n-pose.onnx` zamiast `yolo11n.onnx`. Klasyfikacja aktywnosci oparta o geometrie keypoints (katy kolan, rozstaw kostek, pochylenie tulowia). Progi latwwe do tuningu |

### Uzasadnienie

Zleceniodawca potrzebuje odpowiedzi na pytanie "czym zajmowali sie ludzie" — to jest core value proposition. Poziom 1 i 2 nie daja tej odpowiedzi. Poziom 3 z heurystykami (bez dodatkowego modelu ML) to najlepszy kompromis: wartosciowy raport przy umiarkowanej zlozonosci.

---

## 5. Decyzja 3: Dostarczanie wideo od klienta

To byla najtrudniejsza decyzja. Klient typowo nie bedzie reczne eksportowal plikow z NVR i wrzucal ich do systemu.

### Opcja A: Upload MP4 przez przegladarke

Klient eksportuje z NVR, uploaduje plik.

| | Opis |
|---|---|
| **Zalety** | Najprostsza implementacja po naszej stronie |
| **Wady** | Zly UX — klient musi znalezc nagranie w NVR, wyeksportowac, wrzucic (2-5GB). Wiekszosc klientow nie bedzie tego robic regularnie |
| **Konsekwencje** | Moze dzialac na demo, ale nie na produkcji |

### Opcja B: RTSP pull (nasz system laczy sie do kamery klienta)

Nasz backend laczy sie bezposrednio do kamery klienta po RTSP.

| | Opis |
|---|---|
| **Zalety** | Dobry UX — klient podaje URL kamery, reszta automatyczna |
| **Wady** | Wymaga otwarcia portow / port forwarding na routerze klienta. Duze ryzyko bezpieczenstwa (kamera dostepna z internetu). Problemy z NATem. Wymaga stalego polaczenia |
| **Konsekwencje** | Nierealistyczne bez VPN. Ryzyko bezpieczenstwa nie do zaakceptowania |

### Opcja C: Lekki agent u klienta (outbound push) ✅ WYBRANA

Maly daemon Docker na sieci LAN klienta. Laczy sie lokalnie do kamery, nagrywa, pushuje do R2.

| | Opis |
|---|---|
| **Zalety** | **Najlepszy UX** — agent robi wszystko automatycznie. Zero otwartych portow (outbound HTTPS only). Agent laczy sie do kamery po LAN (brak problemow sieciowych). Identyczny wzorzec jak gpu-agent (pull-based, R2 mediator). Jednorazowa instalacja |
| **Wady** | Klient musi zainstalowac Docker + agent (jednorazowo). Wymaga PC na sieci klienta. Wiecej pracy po naszej stronie (budowa agenta) |
| **Konsekwencje** | Budujemy `client-agent` w Dockerze z prostym UI (Flask :8080). Klient konfiguruje RTSP URL kamery + czas nagrania. Agent nagrywa via ffmpeg, uploaduje chunki do R2. Instruktaz instalacji w formie wideo |

### Opcja D: Integracja z API NVR (Hikvision, Dahua)

Nasz system laczy sie do API konkretnych producentow NVR.

| | Opis |
|---|---|
| **Zalety** | Brak koniecznosci instalacji agenta. Automatyczny dostep do nagran |
| **Wady** | Wymaga otwarcia NVR na internet (port forwarding) — ryzyko bezpieczenstwa. Kazdy producent ma inne API (fragmentacja). Wymaga credentials NVR |
| **Konsekwencje** | Mozliwe jako rozszerzenie agenta (agent na LAN laczy sie do NVR API lokalnie), ale nie jako standalone |

### Opcja E: Cloudflare Tunnel

Klient instaluje `cloudflared`, tuneluje RTSP.

| | Opis |
|---|---|
| **Zalety** | Bezpieczny tunel, brak otwartych portow |
| **Wady** | `cloudflared` tuneluje HTTP/HTTPS, **nie raw TCP/RTSP**. Wymaga konwersji RTSP→HLS na stronie klienta (dodatkowy serwis: mediamtx). Klient musi postawic 2 serwisy. Latencja HLS |
| **Konsekwencje** | Zbyt zlozone na MVP. Efektywnie = opcja C ale z dodatkowa warstwa |

### Uzasadnienie

Opcja C (client-agent) wygrywa bo:
1. **Zero konfiguracji sieciowej** — agent laczy sie do kamery po LAN, do R2 po outbound HTTPS
2. **Identyczny wzorzec jak gpu-agent** — sprawdzony pattern w architekturze projektu
3. **Jednorazowa instalacja** — Docker + `.env` + `docker compose up`
4. **Maksymalnie proste dla klienta** — po instalacji klient widzi prosty UI w przegladarce

Opcje D (NVR API) i E (tunel) moga byc rozszerzeniem agenta w przyszlosci — agent na LAN klienta moze rowniez pobierac nagrania z NVR API lokalnie.

---

## 6. Decyzja 4: Tracking osob

### Opcja A: Agregat per-frame (bez trackingu) — PIERWOTNIE WYBRANA, ZASTĄPIONA

Kazda klatka analizowana niezaleznie: "2 osoby stoja, 1 siedzi". Sumowanie po czasie = osobo-minuty.

| | Opis |
|---|---|
| **Zalety** | Znacznie prostsza implementacja. Nie wymaga ByteTrack/DeepSORT. Brak problemow z re-identyfikacja (osoba znika i wraca do kadru). Wystarczajace do odpowiedzi "ile czasu laczne spedzono na X" |
| **Wady** | Nie wiemy ile unikatowych osob bylo. Nie mozemy powiedziec "Osoba #1 chodzila 15min, Osoba #2 siedziala 30min" |
| **Konsekwencje** | Raport operuje na **osobo-minutach** (person-minutes), nie na per-person breakdown |

### Opcja B: Per-person tracking (ByteTrack + OSNet) ✅ OBECNIE WYBRANA

Sledzenie kazdej osoby miedzy klatkami z unikalnym ID.

| | Opis |
|---|---|
| **Zalety** | Duzo wartosciowszy raport — per-person breakdown. Mozna policzyc unikalne osoby |
| **Wady** | Dodatkowa zlozonosc (tracker, re-ID). Problemy: osoba znika za reglem i wraca = nowy ID. Kamery pod katem = czeste okluzje. Znacznie trudniejsze debugowanie |
| **Konsekwencje** | Realny do dodania post-MVP jako rozszerzenie |

### Uzasadnienie

Dla walidacji koncepcji agregat per-frame wystarczy. Klient widzi "laczne 45 osobo-minut stania" — to juz odpowiada na pytanie "czym sie zajmowali ludzie". Per-person tracking to naturalne rozszerzenie po walidacji MVP.

### Aktualizacja 2026-07-15 — wybrano Opcje B (issue #32)

**Status: Opcja A zastapiona przez Opcje B.** "Naturalne rozszerzenie po walidacji MVP" nadeszlo: platny pilot na jednej gietarce wymaga trackingu.

Co wymusilo zmiane:

1. **Dokladnosc.** Klasyfikacja per-frame myli sie o ±16 s (Film 1) i ±7 s (Film 2) wzgledem anotacji recznej. Zrodla bledu — nadwrazliwy prog chodzenia, aliasing przejsc przy 1 fps, sporadyczne false-positive (wozek/lawka na kolkach czytane jako osoba) — wszystkie wynikaja z braku ciaglosci tozsamosci miedzy klatkami. VLM poprawil dokladnosc *per-frame*, ale nie ma pamieci miedzy klatkami, wiec problemu nie rozwiazal.
2. **Skargi klienta (2026-07-15).** "Wozek z tasma liczony jako osoba" → filtr min-track-length. "Wiecej ludzi niz policzono" → zliczanie osobo-minut per track.
3. **Zone tier (#77).** Wykrywanie obecnosci/nieobecnosci przy stanowisku i rozmow wymaga stabilnych ID w obrebie nagrania — bez tego nie odroznimy pracownika od przechodnia.

Decyzje projektowe (doradztwo Andrew Ng, 2026-05-27):

- **Metryka asocjacji: wyglad (OSNet Re-ID), nie IoU.** Przy 1 fps osoba przemieszcza sie miedzy klatkami za daleko, zeby pokrycie boxow cokolwiek znaczylo; standardowy Kalman tez nie ma tu rozdzielczosci czasowej.
- **Dedykowany model OSNet**, nie reuse cech posrednich YOLO — modele Re-ID sa duzo lepsze w utrzymaniu tozsamosci przy 1 fps.
- **Filtr na lawke-na-kolkach: wylacznie min-track-length**, bez progu confidence YOLO (realna osoba w trudnych klatkach ma 0.62–0.92, nie ma czystego ciecia) i bez klasyfikatora osoba/nie-osoba.
- **Domyslnie dzielic, nie scalac** (`max_track_age_s` = 120 s). Scalenie dwoch osob w jeden track po cichu psuje osobo-minuty; podzial jednej osoby na dwa tracki tylko pokazuje przerwe w obecnosci. Przy raportowaniu czasu ten drugi blad jest znacznie tanszy.

Poza zakresem (osobne tiery produktowe): rozpoznawanie twarzy, re-identyfikacja miedzy nagraniami/dniami/kamerami.

---

## 7. Decyzja 5: Prototyp vs. integracja

### Opcja A: Standalone prototyp — PIERWOTNIE WYBRANA

Osobny katalog (`infra/video-test/`), osobne Docker-compose, R2 jako koordynator (bez bazy danych).

| | Opis |
|---|---|
| **Zalety** | Szybka walidacja. Niezalezny od postepow nad glowna platforma. Latwy do zademonstrowania klientowi. Mozna iterowac bez ryzyka dla reszty systemu |
| **Wady** | Duplikacja niektorych patternow. Pozniej wymaga migracji do platformy |
| **Konsekwencje** | Budujemy 3 komponenty: pipeline (AI), client-agent (Flask), gpu-service (worker). Koordynacja via R2 key conventions (bez bazy danych) |

### Opcja B: Od razu jako problem_type w platformie

Nowy `problem_type: 'surveillance_analysis'` w istniejacym job dispatch.

| | Opis |
|---|---|
| **Zalety** | Brak duplikacji. Od razu zintegrowane z billingiem, dashboardem, itp. |
| **Wady** | Glowna platforma nie jest jeszcze zaimplementowana (design docs only). Blokada — nie mozna walidowac AI bez gotowej platformy |
| **Konsekwencje** | Czekamy na implementacje platformy zanim zwalidujemy koncepcje — zbyt wolne |

### Uzasadnienie

Design docs platformy istnieja, ale **zero kodu jest zaimplementowane**. Czekanie na platforme opozniloby walidacje. Standalone prototyp pozwala udowodnic koncepcje natychmiast. Sciezka integracji jest jasna — po walidacji prototyp staje sie nowym Docker service w `gpu-agent` docker-compose z nowym `problem_type`.

### Aktualizacja 2026-07-17 — integracja platformowa jest aktywna

Pierwotna decyzja przyspieszyła walidację, ale jej warunek „platforma nie ma
kodu” już nie obowiązuje. Obecnie istnieją dwie wspierane ścieżki:

- bare-metal appliance rejestruje się, heartbeat'uje, pobiera konfigurację i
  taski oraz uploaduje przez presigned URL;
- `gpu_service.rest_server` udostępnia kontrakt gpu-agent na porcie 5003 i
  zwraca kanoniczny `result.json`;
- standalone worker R2 + dashboard pozostaje ścieżką kompatybilności i
  diagnostyki, a nie jedyną architekturą produktu.

**Status:** hybryda. Zachowujemy samodzielny worker, ale platform integration
nie jest już „future”.

---

## 8. Decyzja 6: Stack technologiczny

### Python ✅ WYBRANY (prototyp)

| | Opis |
|---|---|
| **Zalety** | Najszybszy do prototypowania ML pipeline. `ultralytics` (YOLO), `onnxruntime-gpu`, `opencv` — dojrzale biblioteki. Istniejacy precedens: `surrogate-serve` jest w Pythonie |
| **Wady** | Glowna platforma jest w TypeScript. Dualnosc jezykowa |
| **Konsekwencje** | Pipeline w Pythonie, docelowo jako Docker service (identycznie jak surrogate-serve). Nie wymaga przepisywania — Python zostaje jako runtime dla tego modelu |

### TypeScript/Node.js (rozwazone, odrzucone)

| | Opis |
|---|---|
| **Zalety** | Spojnosc z reszta projektu (yolo-serve jest w TS) |
| **Wady** | Gorsze ML libs. Wolniejsze prototypowanie. `onnxruntime-node` nie ma tak dobrego wsparcia pose jak `ultralytics` |

| | Opis |
|---|---|
| **Zalety** | Bazuje na istniejacym YOLO infra. `-pose` wariant dodaje 17 keypoints. ONNX = vendor-neutral, GPU acceleration via CUDA EP |
| **Wady** | Wymaga poprawnego CUDA/ORT na Blackwell; silent CPU fallback musi być wykryty po utworzeniu sesji |
| **Konsekwencje** | Aktualny default to stały `yolo11s-pose.onnx` 640×640. CPU fallback jest zabroniony. Nano pozostaje wariantem benchmarkowym. |

### Klasyfikacja aktywnosci: heurystyki geometryczne — PIERWOTNY WYBÓR

| | Opis |
|---|---|
| **Zalety** | Zero dodatkowego trenowania. Interpretowalnosc — mozna wytlumaczyc dlaczego klasyfikator powiedzial "siedzi". Latwwe tunowanie progow |
| **Wady** | Ograniczona dokladnosc (~80% target). Nie rozpozna zlozonych aktywnosci ("obslugiuje maszyne", "podnosi przedmiot") |
| **Konsekwencje** | 4 aktywnosci: siedzi/stoi/chodzi/biega. Jesli 80% niedostateczne → mozna dodac lekki model klasyfikacji (np. MLP na keypoints) bez zmiany reszty pipeline |

### Aktualizacja 2026-07-17 — VLM default, MLP odrzucony przez gate

- Dockerowy default to `vlm`: Qwen2.5-VL-3B dla statycznej postawy +
  displacement dla chodzenia.
- Heurystyka pozostaje wspieranym baseline/rollback.
- Per-person MLP został wytrenowany i zintegrowany opcjonalnie, ale zamrożony
  test dał 62.67% wobec 93.33% VLM. Nie został promowany ani wdrożony.
- Cztery klasy aktywności pozostają bez zmian.

---

## 9. Decyzja 7: Format raportu

### Standalone HTML — PIERWOTNIE WYBRANY

Jeden plik HTML z Chart.js inline i screenshotami jako base64.

| | Opis |
|---|---|
| **Zalety** | Klient otwiera w dowolnej przegladarce. Zero dodatkowego softu. Mozna wyslac mailem. Zawiera wykresy i annotowane klatki. Mozna drukowac |
| **Wady** | Plik moze byc duzy (~5-10MB z 5 keyframes base64). Nie jest interaktywny (brak filtrowania, zoomu) |
| **Konsekwencje** | Raport zawiera: tabele podsumowania, pie chart (% per aktywnosc), timeline (stacked bar, 1-min bins), 5 annotowanych klatek kluczowych |

### Inne rozwazone formaty

| Format | Odrzucony bo |
|--------|-------------|
| PDF | Trudniejszy do generowania z wykresami w Pythonie. HTML latwiejszy |
| JSON API | Wymaga frontendu do wyswietlenia. Zbyt duzo pracy na MVP |
| Dashboard webowy | Wymaga hostowania serwera. Zbyt zlozone na prototyp |

### Aktualizacja 2026-07-17 — `result.json` jest kanoniczny

Platforma renderuje raport natywnie z `result.json` schema 6. JSON zawiera
podsumowanie, timeline, JPEG keyframes, tracking diagnostics, strefy, shift,
presence/work/absence i conversation. Standalone HTML pozostał wyłącznie jako
lokalny/debugowy `--format html`; worker i gpu-agent go nie uploadują.

---

## 10. Decyzja 8: RODO i kwestie prawne

### Ignorowane na prototyp ✅ DECYZJA

| | Opis |
|---|---|
| **Stan** | Przetwarzanie wideo z monitoringu pracownikow to RODO minefield (art. 6, art. 9, ocena skutkow). Na etapie prototypu ignorujemy |
| **Ryzyka** | Brak podstawy prawnej przetwarzania. Brak umowy powierzenia danych. Brak analizy DPIA |
| **Plan na przyszlosc** | Przed produkcja: klient musi miec legalna podstawe monitoringu (regulamin pracy). Umowa powierzenia danych (procesor = my, administrator = klient). DPIA. Opcje: przetwarzanie bez zapisu twarzy, automatyczne blurowanie, retencja wideo = 0 (kasowanie po analizie) |

---

## 11. Flow klienta

### Jednorazowy setup (raz)

```
1. Klient otrzymuje od nas:
   - Link do pobrania client-agent (zip/git)
   - Plik .env z credentials (R2 API token, scoped do bucket surveillance-data)
   - Film instruktazowy (instalacja krok po kroku)

2. Klient instaluje Docker Desktop (Windows) lub Docker Engine (Linux)
   na dowolnym PC w sieci LAN z kamerami

3. Klient rozpakowuje client-agent, kopiuje .env

4. Klient uruchamia:
   docker compose -f docker-compose.client.yml up -d

5. Klient otwiera http://localhost:8080 w przegladarce
   → Widzi prosty formularz
```

### Codzienne uzycie

```
1. Klient otwiera http://localhost:8080

2. Klient wpisuje RTSP URL kamery
   (np. rtsp://admin:haslo@192.168.1.100:554/stream1)
   → Klika "Test Connection" → system weryfikuje polaczenie

3. Klient wybiera czas nagrania:
   "Nagraj od teraz przez [1h / 2h / 4h / 8h]"

4. Klient klika "Start Recording"
   → Agent rozpoczyna nagrywanie via ffmpeg
   → Pasek postepu: "Nagrywanie... 45min/60min"

5. Po zakonczeniu nagrywania:
   → Agent dzieli nagranie na chunki 1h (jesli dluzsze)
   → Agent uploaduje chunki do R2 (pasek postepu uploadu)
   → Agent tworzy status.json z status="pending"
   → UI pokazuje: "Przetwarzanie... oczekiwanie na wynik"

6. Agent polluje status.json co 15s
   → Gdy status="processing": "Analiza w toku... 45%"
   → Gdy status="completed": "Raport gotowy!"

7. Klient klika "Pobierz raport" lub "Otworz raport"
   → Otwiera sie standalone HTML w przegladarce
   → Raport zawiera:
     - Podsumowanie: czas trwania, liczba osob, dominujaca aktywnosc
     - Pie chart: % czasu per aktywnosc (siedzi/stoi/chodzi/biega)
     - Timeline: wykres aktywnosci w czasie (1-minutowe interwaly)
     - 5 kluczowych klatek z annotacjami (bounding boxy, szkielety, etykiety)

8. Klient moze wrocic do /jobs i zobaczyc historie wszystkich analiz
```

### Diagram flow klienta

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Instalacja  │────>│  Konfiguracja│────>│  Nagrywanie  │────>│   Upload     │
│  Docker +    │     │  RTSP URL +  │     │  ffmpeg →    │     │  chunki →    │
│  agent       │     │  czas        │     │  MP4 local   │     │  R2 bucket   │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
     (raz)               (per job)            (auto)                  │
                                                                      ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Otwarcie    │<────│  Download    │<────│  Oczekiwanie │<────│  GPU server  │
│  raportu     │     │  report.html │     │  poll co 15s │     │  przetwarza  │
│  w browser   │     │  z R2        │     │  status.json │     │  wideo       │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 12. Flow inwestora

Inwestor to wlasciciel serwerow GPU. Jego rola w kontekscie analizy wideo:

### Automatyczny flow (zero akcji inwestora)

```
1. Serwer inwestora jest w trybie AI (Ubuntu, gpu-agent aktywny)
   → To jest warunek wstepny — inwestor wczesniej przelaczyl tryb na AI
     lub system automatycznie przelaczyl (brak oplacalnego miningu)

2. GPU service (surveillance-serve) dziala jako Docker container
   → Uruchomiony obok istniejacych serwisow (yolo-serve, surrogate-serve)
   → Nie wymaga akcji inwestora

3. Worker polluje R2 co 10s w poszukiwaniu nowych jobow
   → Gdy znajdzie status="pending":
     a. Pobiera wideo z R2 → local temp
     b. Przetwarza: ffmpeg → YOLO-pose → heurystyki → raport
     c. Uploaduje raport do R2
     d. Aktualizuje status="completed"

4. Inwestor widzi w dashboardzie (docelowo):
   → Serwer status: "AI Active — surveillance job in progress"
   → GPU utilization: ~80-90% (YOLO-pose inference)
   → Job history: lista przetworzonych analiz z czasem i kosztami
```

### Co inwestor moze robic (docelowo, po integracji z platforma)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Investor Dashboard                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Serwer: GPU-PL-01          Status: AI Active                   │
│  Tryb: AI (Ubuntu)          Uptime: 99.2%                       │
│  GPU Load: 87%              Temp: 62°C                          │
│                                                                   │
│  ┌─ Aktywny job ──────────────────────────────────────────┐     │
│  │ Job: surveillance_analysis                              │     │
│  │ Postep: 67% (2412/3600 klatek)                         │     │
│  │ Status: processing                                      │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌─ Przychod ─────────────────────────────────────────────┐     │
│  │ Dzisiaj: 12.50 PLN (2.5 GPU-h × 5 PLN/GPU-h)         │     │
│  │ Ten miesiac: 287.00 PLN                                 │     │
│  │ Stawka surveillance: 5 PLN/GPU-h                        │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  [Przelacz na Mining]  [Historia jobow]  [Ustawienia]           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Przelaczanie trybow (mining ↔ AI)

```
Scenariusz: Inwestor chce wrocic do miningu

1. Inwestor klika "Przelacz na Mining" w dashboardzie

2. System sprawdza: czy jest aktywny job?
   → TAK: "Job w toku (67%). Poczekac na zakonczenie?"
     → Inwestor potwierdza → system czeka na zakonczenie joba
     → Lub: inwestor wymusza → job re-queued na inny serwer
   → NIE: przelaczenie natychmiastowe

3. gpu-agent wykonuje:
   sudo grub-reboot 2 && sudo reboot
   → Serwer restartuje do HiveOS (~3-5 min)
   → Mining startuje automatycznie

4. Dashboard: status zmienia sie na "Mining"
   → Brak heartbeatu AI (HiveOS nie ma gpu-agenta)
   → Przychod z miningu widoczny via HiveOS API
```

### Scenariusz: system potrzebuje GPU do analizy wideo

```
1. Klient submituje job → system szuka dostepnego serwera

2. Priorytet alokacji:
   a. Serwer w trybie AI Active (Docker running) → natychmiast
   b. Serwer w trybie AI Standby (Ubuntu, Docker off) → ~30-60s
   c. Serwer w trybie Mining → NIE w MVP (wymaga reboot 3-5min + zgoda inwestora)

3. Jesli brak serwerow AI → job czeka w kolejce
   → Klient widzi: "Oczekiwanie na dostepny serwer..."
   → Docelowo: system moze poprosic inwestora o przelaczenie na AI
```

---

## 13. Podsumowanie decyzji

| # | Decyzja | Wybor | Glowne uzasadnienie |
|---|---------|-------|---------------------|
| 1 | Silnik analizy | Custom pipeline (YOLO-pose) | Frigate = live monitoring, nie batch. Custom = kompatybilny z architektura |
| 2 | Glebokosc raportu | Klasyfikacja aktywnosci (poziom 3) | Core value proposition — "czym sie zajmowali ludzie" |
| 3 | Dostarczanie wideo | Client-agent (outbound push) | Zero konfiguracji sieciowej, sprawdzony wzorzec, najlepszy UX |
| 4 | Tracking | ByteTrack + OSNet, split-over-merge | Stabilne ID i filtr false positives są wymagane przez zone tier |
| 5 | Prototyp vs platforma | Platform REST + zachowany standalone R2 worker | Platform dispatch działa; standalone pozostaje kompatybilnością |
| 6 | Stack | Python + YOLO11s-pose ONNX + VLM | CUDA pipeline i Blackwell są zweryfikowane; VLM wygrał quality gate |
| 7 | Format raportu | `result.json` schema 6; HTML debug-only | Platforma odpowiada za rendering, dane pozostają wersjonowane |
| 8 | RODO | Ignorowane na prototyp | Walidacja technologii first, legal pozniej |

---

## 14. Ryzyka i ograniczenia

| Ryzyko | Wplyw | Mitygacja |
|--------|-------|-----------|
| Brak CUDA EP / silent CPU fallback | Pipeline nie startuje lub łamie budżet | `ort.preload_dlls`, jawny cublas, kontrola providerów i fail-fast; brak CPU fallback |
| MLP szybszy, ale jakościowo słabszy | Regresja raportów po pozornie „tanim” modelu | Zamrożony promotion gate; VLM pozostaje defaultem |
| Kamera pokazuje za mało pikseli na pracownika | Żaden software arm nie daje wiarygodnego zone report | #86 opublikował no-winner; #88 wymaga station-framed streamu |
| Długi czas przetwarzania | Opóźnienie raportu zależne od fixture | Nie podawać ETA bez kalibracji; używać zmierzonych artifactów i jawnej ekstrapolacji |
| Wolny upload klienta | Opóźnienie dostarczenia wideo | Rolling buffer + presigned chunk upload; mierzyć na docelowym łączu |
| Klient nie ma Dockera lub sudo | Trudne wdrożenie | Bare-metal root installer oraz user-mode systemd z weryfikacją linger |
| Okluzje i niepewna re-identyfikacja | Split/merge tracków | Preferować split; nie scalać automatycznie po długich przerwach; brak face recognition |
| RODO — przetwarzanie wizerunku pracownikow | Ryzyko prawne | Przed produkcja: umowa powierzenia, DPIA, opcja auto-blur twarzy |
