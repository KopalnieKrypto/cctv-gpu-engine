# Plan: Client-Agent Standalone Appliance

> Source PRD: rozmowa projektowa z 2026-05-05 (kontekst w pamięci: `project_client_agent_dual_target.md`).
> Cel: standalone appliance dla mini-PC (Pi 5 / Intel N100) w LAN, z auto-discovery kamer i credentiali z plików env. Docker stack pozostaje nietknięty.

## Architectural decisions

Trwałe decyzje, które obowiązują we wszystkich fazach:

- **Architektura kodu**: jeden pakiet `client_agent/` zawiera całą logikę biznesową (recorder, r2_client, web, discovery). **Dwa entrypointy** w tym samym pakiecie: `client_agent.agent` (Docker, istniejący) i `client_agent.appliance` (standalone, nowy). Osobny katalog `client-appliance/` zawiera **wyłącznie packaging** — systemd unit, env templates, install.sh, README. Zero kodu Pythona.
- **Nieusuwalność Dockera**: Docker stack (`client-agent/Dockerfile`, `docker-compose.client.yml`) **pozostaje bez zmian**. Każda nowa funkcja (np. discovery) musi działać też w Dockerze — wspólny pakiet to gwarantuje.
- **Discovery**: ONVIF WS-Discovery (multicast UDP 239.255.255.250:3702) jako podstawa. Hikvision SADP fallback **odłożony** na osobne issue — dodajemy tylko jeśli ONVIF nie złapie realnych kamer. Discovery uruchamiane **on-demand z UI**, nie cyklicznie, nie przy starcie usługi.
- **UI**: Flask na :8080 zostaje (bez zmian topologii). Nowy endpoint `/cameras/discover` + przycisk w istniejącym `index.html`.
- **Tryb pracy appliance**: jednorazowe nagranie wywoływane z UI (jak w Docker dzisiaj). Brak harmonogramu, brak ciągłego nagrywania.
- **Credentiale**:
  - Kamery: wspólny default user/pass + opcjonalny override per IP (`RTSP_DEFAULT_USER`, `RTSP_DEFAULT_PASS`, `RTSP_CAM_<ip>_USER`, `RTSP_CAM_<ip>_PASS`) w `cameras.env`.
  - R2: osobny `r2.env` z `R2_*` keys.
  - Wczytywane przez systemd `EnvironmentFile=` z `/etc/cctv-client/`.
- **Storage**: `RECORDINGS_DIR` defaultuje na XDG state dir (`${XDG_STATE_HOME:-$HOME/.local/state}/cctv-client/recordings`) — odporne na `PrivateTmp=yes` w systemd. Docker dalej używa `/tmp/cctv-recordings` przez env override.
- **WSGI server (appliance)**: waitress zamiast Werkzeug dev server. Wielowątkowy, czysty Python, jeden dodatkowy dep.
- **Sprzęt docelowy**: Pi 5 8GB / Intel N100 mini-PC z NVMe; 1-8 kamer; nagrania do 8 h. Bez GPU.
- **Integracje zewnętrzne**: bez zmian — boto3 → R2 bucket `surveillance-data`, praca podejmowana przez gpu-service na VPS.

---

## Phase 1: ONVIF discovery + endpoint w UI

**User stories**: Operator otwiera Flask UI :8080, klika "Wykryj kamery" i widzi listę wykrytych kamer ONVIF w LAN — vendor, model, IP, port, gotowy RTSP URL (bez credentiali jeszcze).

### What to build

End-to-end vertical slice działający **w istniejącym kontenerze Docker** (zanim ruszymy packaging). Nowy moduł discovery w pakiecie `client_agent` używa WS-Discovery do znalezienia kamer ONVIF, dla każdej probuje GetServices/GetStreamUri, wraca listę. Nowy endpoint Flask serwuje wynik jako JSON. Istniejący `index.html` dostaje przycisk "Wykryj kamery" + sekcję z listą wyników. Klik na pozycji listy wkleja RTSP URL do istniejącego formularza nagrywania.

To jest tracer bullet weryfikujący **jedyne nieznane techniczne ryzyko** projektu: czy ONVIF złapie realne kamery operatora. Jeśli nie zadziała na ≥1 kamerze, Phase 1 trzeba poszerzyć o SADP zanim ruszymy dalej.

### Acceptance criteria

- [ ] `client_agent/discovery.py` eksportuje funkcję wracającą listę dataclass `DiscoveredCamera{ip, port, vendor, model, rtsp_url}`
- [ ] WS-Discovery działa z timeoutem ≤ 5 s i nie blokuje request thread Flask poza ten limit
- [ ] Endpoint `GET /cameras/discover` zwraca JSON `{cameras: [...], scanned_at: <iso>, error: null|string}`
- [ ] Discovery działa wewnątrz `cctv-client-agent` kontenera (Docker bridge network — multicast może wymagać `network_mode: host` lub konfiguracji; jeśli tak, udokumentowane w komentarzu w docker-compose.client.yml)
- [ ] Przycisk "Wykryj kamery" w UI; wyniki renderują się jako klikalna lista; klik wkleja RTSP URL do pola formularza nagrywania
- [ ] Test jednostkowy z mockowanym ONVIF response (bez sieci) — kontrakt parsowania
- [ ] Manualny test na ≥ 1 realnej kamerze IP w LAN — discovery zwraca poprawny RTSP URL
- [ ] Brak regresji: istniejący flow ręcznego wpisywania RTSP URL nadal działa

---

## Phase 2: Credentiale z env + auto-budowanie RTSP URL

**User stories**: Operator klika wykrytą kamerę z listy → wybiera czas nagrania → klika "Nagraj". Recorder pobiera user/pass z konfiguracji (default lub per-IP override) i samodzielnie składa pełny RTSP URL z credentialami. Operator nie wpisuje hasła w UI.

### What to build

Warstwa rozwiązywania credentiali w pakiecie `client_agent`. Czyta zmienne środowiskowe (lub plik env z lokalnej ścieżki w trybie standalone — w Dockerze zostaje czyste env). Gdy `web.py` dostaje request "nagraj kamerę z IP X", pyta credentials resolver o user/pass dla tego IP, składa RTSP URL `rtsp://user:pass@ip:port/path` i przekazuje do istniejącego `Recorder.start()`. Wszystko inne (segmentacja, upload do R2, status.json) zostaje bez zmian.

UI dostaje drugi tryb startu nagrania: "z wykrytej kamery" (tylko IP + duration) obok istniejącego "z ręcznym URL".

### Acceptance criteria

- [ ] Resolver credentiali w `client_agent` z hierarchią: env per-IP (`RTSP_CAM_<sanitized_ip>_USER/_PASS`) → env default (`RTSP_DEFAULT_USER/_PASS`) → błąd 400 z czytelnym komunikatem
- [ ] Sanityzacja IP do nazwy zmiennej udokumentowana (kropki → podkreślenia)
- [ ] Endpoint `POST /start` akceptuje alternatywny format body `{camera_ip, duration_s}` obok istniejącego `{rtsp_url, duration_s}`
- [ ] UI: po kliknięciu wykrytej kamery formularz dostaje `camera_ip` zamiast pełnego URL; hasło nigdy nie pojawia się w DOM ani logach Flask
- [ ] Test jednostkowy: hierarchia rozwiązywania credentiali (per-IP wygrywa z default; brak default → error)
- [ ] Manualny e2e w Dockerze: `cameras.env` z defaultowym user/pass, kamera wykryta przez Phase 1, klik "Nagraj 30 s", plik w R2 ma poprawną zawartość
- [ ] Logi Flask **nie zawierają** hasła w żadnej formie (test: grep logu po nagraniu)

---

## Phase 3: Standalone entrypoint + state dir + waitress

**User stories**: Operator uruchamia appliance bezpośrednio na mini-PC bez Dockera (`python -m client_agent.appliance`) i ma 1:1 funkcjonalność Phase 1+2 — discovery, nagrywanie z env credentialami, upload do R2.

### What to build

Drugi entrypoint w pakiecie `client_agent`. Inicjalizuje WSGI server (waitress, wielowątkowy), czyta env z plików w lokalizacji konfigurowalnej flagą CLI (`--env-dir`, default `/etc/cctv-client`), defaultuje `RECORDINGS_DIR` na XDG state dir z fallbackiem na `~/.local/state/cctv-client/recordings`, montuje tę samą Flask app co Docker entrypoint. Drobne fixy w istniejącym kodzie żeby ścieżki nie zakładały kontenera (np. brak hardcoded `/tmp`).

Zmiany w istniejącym `agent.py` minimalne — tylko ekstrakcja konstrukcji Flask app do reużywalnej funkcji. Docker entrypoint nadal działa identycznie.

### Acceptance criteria

- [ ] `python -m client_agent.appliance` startuje waitress na :8080 i serwuje wszystkie istniejące endpointy
- [ ] Flag `--env-dir <path>` (default `/etc/cctv-client`) wczytuje `cameras.env` i `r2.env` przy starcie
- [ ] `RECORDINGS_DIR` defaultuje na XDG state dir; tworzony idempotentnie przy starcie
- [ ] Docker entrypoint (`python -m client_agent.agent`) nadal działa bez zmian — smoke test compose stack
- [ ] Brak nowych systemowych dependencies poza tymi już wymaganymi (ffmpeg, Python 3.12)
- [ ] Manualny e2e na laptopie/Pi (poza Dockerem): venv → `uv sync` → `python -m client_agent.appliance` → discovery + nagrywanie + upload działają
- [ ] Dokumentacja: w `client-agent/README.md` (lub nowy `client-appliance/README.md`) sekcja "Standalone (poza Dockerem)" z przykładem CLI
- [ ] CLI ma flagę `--foreground` / domyślny tryb foreground odpowiedni dla `Type=simple` w systemd

---

## Phase 4: Packaging + systemd + install.sh

**User stories**: Operator dostaje tarball z appliance, na świeżym mini-PC uruchamia `install.sh` jako root, po reboot urządzenie samo wstaje, Flask UI dostępne w LAN, nagrywanie z UI ląduje w R2 bez żadnej dodatkowej interwencji.

### What to build

Nowy katalog `client-appliance/` w repo zawierający wyłącznie packaging — żadnego kodu Pythona. systemd unit `cctv-client.service` (Type=simple, After=network-online.target, EnvironmentFile=/etc/cctv-client/r2.env i cameras.env, ExecStart wskazuje na venv w `/opt/cctv-client`). Skrypt `install.sh` idempotentny — tworzy usera systemowego `cctv`, venv w `/opt/cctv-client`, kopiuje pakiet `client_agent`, instaluje deps przez uv, tworzy `/etc/cctv-client/` z permissions 600 i przykładowymi env, instaluje unit, `systemctl enable --now`. Pliki przykładowe `cameras.env.example` i `r2.env.example` z komentarzami. README z procedurą instalacji, troubleshootingiem (multicast w LAN, firewall :8080) i instrukcją update.

Smoke test na świeżym Linuksie (VM lub realny mini-PC): od `tar xzf` do działającego UI w LAN ≤ 5 minut.

### Acceptance criteria

- [ ] `client-appliance/install.sh` idempotentny — drugie uruchomienie nie psuje istniejącej instalacji
- [ ] systemd unit przeżywa reboot — `systemctl status cctv-client` po reboot zwraca active
- [ ] `EnvironmentFile=` pobiera oba env (`r2.env`, `cameras.env`); permission 600; właściciel root lub `cctv`
- [ ] Logi service idą do journald i są czytelne przez `journalctl -u cctv-client`
- [ ] `client-appliance/README.md` zawiera: wymagania sprzętowe (link do tabeli z konwersacji), procedurę instalacji, sekcję troubleshooting (ONVIF nie wykrywa → multicast w LAN; UI nieosiągalne → firewall), procedurę update (pull repo + reinstall)
- [ ] Tarball release-ready — opcjonalnie target `make package-appliance` budujący archiwum z venv-em lub bez (do decyzji w trakcie)
- [ ] Smoke test runbook: świeży Ubuntu 24.04 LTS lub Raspberry Pi OS, od `tar xzf` do nagrania w R2 ≤ 5 min, udokumentowany w README
- [ ] Brak regresji w Docker stacku — `docker compose -f docker-compose.client.yml up` nadal działa
