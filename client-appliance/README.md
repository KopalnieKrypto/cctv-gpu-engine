# cctv-client appliance

Standalone packaging dla `client_agent` — Flask UI :8080 + RTSP recorder z
auto-discovery kamer ONVIF, działający bez Dockera na mini-PC w LAN. To
**jedyny** sposób uruchomienia klienta (Dockerowy client-agent został
wycofany w #29). Cała logika żyje w pakiecie
[`client_agent`](../client-agent/client_agent/); ten katalog zawiera
**wyłącznie packaging**: systemd unit, install.sh, env templates, README.

## Wymagania sprzętowe

- Mini-PC z Linuksem: **Raspberry Pi 5 8 GB** lub **Intel N100 (z NVMe)** —
  zgodnie z `plans/client-appliance.md` § Architectural decisions.
- 1–8 kamer IP z ONVIF (do nagrań do 8 h ciągłych).
- Linux: Ubuntu 24.04 LTS lub Raspberry Pi OS (Bookworm) — Python 3.12,
  systemd, ffmpeg dostępne w repo.
- Sieć: kamery i urządzenie w jednym L2 (multicast UDP 239.255.255.250:3702
  używany przez ONVIF discovery — patrz Troubleshooting).
- Brak GPU (appliance robi tylko nagrywanie + upload; analiza działa po
  stronie `gpu-service`).

## Decyzja architektoniczna: git clone vs tarball

Wybór: **git clone jako podstawowa ścieżka instalacji**, tarball jako opcja
dla setupów offline.

Rationale:
- mini-PC w LAN i tak ma dostęp do sieci (R2 + ONVIF), więc `git pull`
  podczas update nie jest barierą;
- artefakt jest mały (~kilka MB), nie ciągniemy 200 MB venv-a;
- update = `git pull && sudo ./client-appliance/install.sh` — krótkie,
  powtarzalne, audytowalne (commit hash mówi co jest zainstalowane).

Tarball pozostaje opcją dla operatorów bez dostępu do GitHuba na
urządzeniu — patrz sekcja "Update" niżej.

## Instalacja

Wymagane na świeżym systemie:

```bash
sudo apt-get update
sudo apt-get install -y git python3.12-venv ffmpeg
```

Sklonuj repo i uruchom installer (jako root):

```bash
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git /opt/src/cctv-gpu-engine
cd /opt/src/cctv-gpu-engine
sudo ./client-appliance/install.sh
```

Po zakończeniu wpisz dane RTSP (i opcjonalnie dane platformy) do plików
konfiguracyjnych — klient nie trzyma już żadnych credentiali R2:

```bash
sudo nano /etc/cctv-client/cameras.env    # RTSP_DEFAULT_USER/PASS (+ opcjonalne per-IP)
sudo nano /etc/cctv-client/platform.env   # PLATFORM_URL + APPLIANCE_TOKEN (tryb platformowy)
sudo systemctl restart cctv-client
```

UI dostępne na `http://<ip-urządzenia>:8080/`.

`install.sh` jest idempotentny — możesz go uruchomić ponownie po pull-u
nowej wersji repo i nie nadpisze plików w `/etc/cctv-client/`.

## Instalacja user-mode (bez roota) — issue #84

Jeśli na urządzeniu **nie masz hasła sudo**, użyj `install-user.sh` zamiast
`install.sh`. Wybór jest binarny:

| Masz roota / hasło sudo? | Skrypt | Layout | Unit |
|---|---|---|---|
| **Tak** (domyślnie) | `sudo ./client-appliance/install.sh` | user `cctv`, `/opt/cctv-client`, `/etc/cctv-client` | `/etc/systemd/system/cctv-client.service` (`WantedBy=multi-user.target`) |
| **Nie** | `./client-appliance/install-user.sh` (**nie** przez sudo) | wszystko w `$HOME`: `~/.local/share/cctv-client`, `~/.config/cctv-client` | `~/.config/systemd/user/cctv-client.service` (z `cctv-client-user.service`, `WantedBy=default.target`) |

Preferuj wariant root, jeśli masz wybór — jeden appliance na maszynę, config
w `/etc`, standardowe `systemctl status`. Wariant user-mode istnieje dla
sytuacji, w której root jest niedostępny operacyjnie.

```bash
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git ~/cctv-gpu-engine
cd ~/cctv-gpu-engine
./client-appliance/install-user.sh          # BEZ sudo — skrypt odmówi startu jako root
nano ~/.config/cctv-client/cameras.env
nano ~/.config/cctv-client/platform.env
systemctl --user restart cctv-client
```

Komendy operatorskie mają **`--user`** (unit żyje w user managerze, nie w
systemowym):

```bash
systemctl --user status cctv-client
systemctl --user restart cctv-client
journalctl --user -u cctv-client -f
```

### Linger — warunek konieczny startu po reboocie

Unit `--user` startuje przy boocie **tylko** gdy dla użytkownika włączony jest
**linger**. Bez niego systemd ubija user managera (i appliance razem z nim) w
momencie zakończenia ostatniej sesji SSH — appliance przeżyje `exit`, ale
zniknie po reboocie, **cicho**. To dokładnie ten tryb awarii, który kosztował
~18 h przestoju na `cameraboy` (2026-07-14 → 2026-07-16).

`install-user.sh` włącza linger sam i **weryfikuje**, że się udało (jeśli nie —
kończy się błędem, zamiast zostawić appliance, który działa do pierwszego
restartu):

```bash
loginctl enable-linger "$(id -un)"
loginctl show-user "$(id -un)" --property=Linger   # oczekiwane: Linger=yes
```

**Czy to wymaga roota?** Zweryfikowane na `cameraboy` (Ubuntu 24.04, systemd
255): **nie**. Akcja polkit `org.freedesktop.login1.set-self-linger` ma
domyślnie `allow_any=yes`, więc użytkownik włączający linger *sobie samemu*
nie przechodzi żadnej autoryzacji. Uwaga: dotyczy to wyłącznie wariantu "sobie
samemu" — podanie **cudzej** nazwy użytkownika trafia w
`set-user-linger`, które autoryzacji wymaga.

Jeśli na innym site'cie polityka polkit jest zaostrzona i `install-user.sh`
zakończy się błędem `linger is still disabled`, potrzebna jest **jednorazowa**
akcja admina (`sudo loginctl enable-linger <user>`) — nadal nieporównanie mniej
niż pełny layout roota.

### Czego user unit nie ma (i dlaczego)

- **Brak `User=`/`Group=`** — unit `--user` z definicji działa jako właściciel;
  systemd odrzuca te dyrektywy w user managerze.
- **Brak `After=/Wants=network-online.target`** — ten target istnieje tylko w
  managerze systemowym (`systemctl --user show network-online.target` →
  `LoadState=not-found`). Skopiowanie go z root unita nic nie daje: `Wants=` na
  nieistniejącym unicie to słaba zależność, więc systemd tylko ostrzeże i
  pojedzie dalej — błąd przeżyłby review i niczego nie kupił. Appliance i tak
  retry'uje, a twarde wyjście łapie `Restart=on-failure`.
- **Reszta jest 1:1 z root unitem**: `Type=simple`, `Restart=on-failure`,
  `RestartSec=5`, logi do journalda.

### Migracja z ręcznego `nohup` (site `cameraboy`)

Site'y bootstrapowane przed #84 mają checkout w `~/cctv-gpu-engine`, env dir w
`~/cctv-client-<site>-production` i proces odpalony z ręki. Migracja:

```bash
pkill -f "client_agent.appliance"                      # ubij ręczny proces
mkdir -p ~/.config/cctv-client
cp ~/cctv-client-*-production/*.env ~/.config/cctv-client/   # przenieś creds
cd ~/cctv-gpu-engine && git pull --ff-only
./client-appliance/install-user.sh
systemctl --user status cctv-client
```

`install-user.sh` kopiuje `client_agent` do site-packages venva, więc unit
**nie potrzebuje** `PYTHONPATH` — w przeciwieństwie do starego launchera, który
działał wyłącznie dzięki `PYTHONPATH` eksportowanemu w shellu operatora
(systemd unit nie dziedziczy środowiska shella).

## Update

Aktualizacja do nowej wersji:

```bash
cd /opt/src/cctv-gpu-engine
sudo git pull --ff-only
sudo ./client-appliance/install.sh
sudo systemctl restart cctv-client
```

`install.sh` przesynchronizuje deps w venv, podmieni unit file (jeśli się
zmienił), zostawi `/etc/cctv-client/*.env` nietknięte. Kontrola wersji:
`git -C /opt/src/cctv-gpu-engine rev-parse HEAD`.

**Wariant offline (tarball)**: na hoście z dostępem do internetu zrób
`tar czf cctv-client-$(git rev-parse --short HEAD).tar.gz cctv-gpu-engine/`,
przenieś plik na docelowy mini-PC, rozpakuj do `/opt/src/`, uruchom
`install.sh`. Ten sam skrypt — różnica tylko w transporcie.

## Troubleshooting

### ONVIF discovery nic nie znajduje

- **Multicast w LAN**: discovery używa WS-Discovery (UDP multicast
  239.255.255.250:3702). Router z włączonym IGMP snooping bez aktywnej
  grupy zablokuje pakiety. Sprawdź `tcpdump -i <iface> udp port 3702` na
  urządzeniu — jeśli nie widać requestów wracających, problem jest na
  switch'u/routerze.
- **VLAN/L3**: kamery w innym VLAN-ie niż appliance — discovery nie
  przechodzi. Wpisz RTSP URL ręcznie w UI (formularz "z ręcznym URL")
  zamiast polegać na auto-discovery.
- **Hikvision bez ONVIF**: część modeli ma ONVIF wyłączony fabrycznie.
  Włącz w panelu kamery (Network → Advanced → Integration Protocol).

### UI nieosiągalne na :8080

- **Firewall (UFW/nftables)**: `sudo ufw allow 8080/tcp` lub odpowiednik.
  Ubuntu Server 24.04 ma UFW domyślnie wyłączony, ale RPi OS i custom
  obrazy mogą blokować.
- **Service nie wstał**: `systemctl status cctv-client` i
  `journalctl -u cctv-client -n 100`. W trybie platformowym częsta
  przyczyna to zła wartość `BUFFER_HOURS` w `platform.env` (nie-liczbowa
  lub `≤ 0` → boot validation fails fast). Brak `PLATFORM_URL`/
  `APPLIANCE_TOKEN` nie wywala unit-a — appliance auto-fallbackuje do
  trybu standalone.
- **Bind na 0.0.0.0**: appliance bindu je do `0.0.0.0:8080` — jeśli mini-PC
  ma kilka interfejsów (np. WiFi + LAN), UI będzie na każdym z nich.

### Nagrywanie się rozjeżdża / 401 Unauthorized

Kamera odrzuca creds — sprawdź hierarchię:

1. `RTSP_CAM_<sanitized_ip>_USER/PASS` w `cameras.env` (kropki w IP →
   podkreślenia, np. `192.168.50.2` → `RTSP_CAM_192_168_50_2_USER`).
2. fallback: `RTSP_DEFAULT_USER/PASS`.

`journalctl -u cctv-client | grep -i 401` pokaże, którą kamerę odrzuca.

## Smoke test (5 min na świeżym systemie)

Runbook weryfikujący end-to-end na świeżym Ubuntu 24.04 LTS lub Raspberry
Pi OS:

1. `sudo apt-get install -y git python3.12-venv ffmpeg` (≤ 30 s).
2. `git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git
   /opt/src/cctv-gpu-engine` (≤ 30 s).
3. `sudo /opt/src/cctv-gpu-engine/client-appliance/install.sh` (≤ 2 min,
   pierwszy run instaluje deps).
4. Edytuj `/etc/cctv-client/cameras.env` (default RTSP user/pass; opcjonalnie
   `platform.env` dla trybu platformowego), `sudo systemctl restart cctv-client`.
5. W przeglądarce: `http://<ip>:8080/` → "Wykryj kamery" → wybierz
   kamerę → "Nagraj 30 s" → sprawdź nagrany chunk w lokalnym buforze
   (recorder jest buffer-only; w trybie platformowym chunk trafia do R2
   przez presigned URL).

Cel: krok 1–5 ≤ 5 min. Jeśli przekracza, problem zwykle leży w sieci
(multicast, firewall) i opisany jest w sekcji Troubleshooting.

## Pliki w tym katalogu

| Plik | Opis |
|------|------|
| `cctv-client.service` | systemd unit — **root** (Type=simple, EnvironmentFile dla `cameras.env` i `platform.env`, ExecStart na venv `/opt/cctv-client`, `WantedBy=multi-user.target`) |
| `cctv-client-user.service` | systemd unit — **user-mode** (#84; bez `User=`, ścieżki przez `%h`, `WantedBy=default.target`) |
| `install.sh` | idempotentny installer **root** (user `cctv`, venv, deps via uv/pip, kopiowanie pakietów, etc) |
| `install-user.sh` | idempotentny installer **user-mode** (#84; venv + config w `$HOME`, linger, `systemctl --user`) |
| `cameras.env.example` | template dla `/etc/cctv-client/cameras.env` (RTSP_DEFAULT_USER/PASS + opcjonalne per-IP) |
| `platform.env.example` | template dla `/etc/cctv-client/platform.env` (PLATFORM_URL, APPLIANCE_TOKEN, opcjonalny BUFFER_HOURS) |

## Tryby pracy: standalone vs platform mode (issue #30)

Po wprowadzeniu integracji z GPU Exchange (`PLATFORM_URL`/`APPLIANCE_TOKEN`)
appliance ma dwa równoległe tryby pracy. Oba uruchamiane są tym samym
entrypointem `client_agent.appliance` (bare-metal, systemd) — różnią się
wyłącznie zawartością env-files w `/etc/cctv-client/`.

| Tryb | Env | Zachowanie |
|------|-----|------------|
| **Standalone** | `cameras.env` (`RTSP_*`), `platform.env` pusty | Flask UI :8080, camera discovery + snapshoty, recorder buffer-only (rolling buffer na dysku, bez uploadu) |
| **Platform mode** (Phase 4) | `PLATFORM_URL` + `APPLIANCE_TOKEN` + opcjonalny `BUFFER_HOURS` w `platform.env` | Flask UI :8080 + TaskPoller w tle, rejestracja w platformie, rolling buffer, upload przez presigned URL |

Wybór trybu zależy od tego, czy appliance jest podpięty do GPU Exchange:

- **Standalone (bez platformy)**: ustaw tylko `RTSP_*` w `/etc/cctv-client/cameras.env`. `platform.env` zostaw pusty — appliance działa w trybie standalone, buffer-only (`_is_platform_mode()` zwraca `False`).
- **Z platformą**: wpisz `PLATFORM_URL` + `APPLIANCE_TOKEN` (+ opcjonalnie `BUFFER_HOURS`) w `/etc/cctv-client/platform.env`. Po restarcie unit-a appliance zarejestruje się, podejmie task'i z kolejki i będzie utrzymywał rolling buffer.

### `BUFFER_HOURS` — rozmiar rolling buffera

`BUFFER_HOURS` w `platform.env` kontroluje retencję per-camera bufora:

- domyślnie `1` (dev / MVP demo) — wystarczy do krótkich task'ów rzędu kilku minut wstecz;
- produkcja: ustaw `8` (lub więcej) — operator zlecający task forensic z 6-godzinnym horyzontem musi mieć w buforze odpowiedni materiał;
- wartość nie-liczbowa lub `≤ 0` → boot validation fails fast (`systemctl status cctv-client` pokaże stack ze stringa `BUFFER_HOURS must be ...`).

### Migracja legacy Docker → bare-metal (issue #29 — **DONE**)

Issue [#29](https://github.com/KopalnieKrypto/cctv-gpu-engine/issues/29)
**wycofał** legacy tryb Dockerowy `client_agent.agent`. Usunięte zostały
`client-agent/Dockerfile`, `docker-compose.client.yml`,
`docker-compose.appliance.yml`, `.env.client.example` oraz
`client_agent/r2_client.py`; obraz GHCR `client-agent` nie jest już
budowany. `client_agent.agent` **nie jest już entrypointem** — pozostaje
wyłącznie jako fabryka `build_app` importowana przez appliance.

**Nota upgrade dla istniejących wdrożeń** (operatorzy podnoszący starszą
instalację):

- (a) legacy ręczny on-site UI ("upload MP4 / nagraj → R2") został wycofany —
  trasy `/upload`, `/start`, `/jobs`, `/report` zwracają teraz **503**. UI
  nadal działa dla: discovery kamer, snapshotów per-kamera, panelu managed
  cameras, `/test-connection` i `/stop`.
- (b) w platform mode appliance uploaduje **wyłącznie przez presigned URL** —
  nie ma już żadnego bezpośredniego dostępu do R2 z klienta.
- (c) usuń pozostały `r2.env` z `/etc/cctv-client/` — jest teraz ignorowany;
  jego usunięcie oznacza **zero credentiali R2 na dysku** klienta.
- (d) nie ma już żadnego wariantu Dockerowego klienta — używaj wyłącznie
  instalacji bare-metal przez systemd (`sudo ./client-appliance/install.sh` +
  `systemctl enable --now cctv-client`).
