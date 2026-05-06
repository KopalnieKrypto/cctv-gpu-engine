# cctv-client appliance

Standalone packaging dla `client_agent` — Flask UI :8080 + RTSP recorder z
auto-discovery kamer ONVIF, działający bez Dockera na mini-PC w LAN. Cała
logika żyje w pakiecie [`client_agent`](../client-agent/client_agent/) i jest
współdzielona z obrazem Dockerowym; ten katalog zawiera **wyłącznie
packaging**: systemd unit, install.sh, env templates, README.

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

Po zakończeniu wpisz dane R2 i RTSP do plików konfiguracyjnych:

```bash
sudo nano /etc/cctv-client/r2.env
sudo nano /etc/cctv-client/cameras.env
sudo systemctl restart cctv-client
```

UI dostępne na `http://<ip-urządzenia>:8080/`.

`install.sh` jest idempotentny — możesz go uruchomić ponownie po pull-u
nowej wersji repo i nie nadpisze plików w `/etc/cctv-client/`.

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
  `journalctl -u cctv-client -n 100`. Najczęstsza przyczyna: brakujące
  klucze w `r2.env` (`R2_ENDPOINT` / `R2_ACCESS_KEY_ID` /
  `R2_SECRET_ACCESS_KEY` są wymagane).
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
4. Edytuj `/etc/cctv-client/r2.env` (R2 creds) i `cameras.env` (default
   RTSP user/pass), `sudo systemctl restart cctv-client`.
5. W przeglądarce: `http://<ip>:8080/` → "Wykryj kamery" → wybierz
   kamerę → "Nagraj 30 s" → sprawdź job w R2.

Cel: krok 1–5 ≤ 5 min. Jeśli przekracza, problem zwykle leży w sieci
(multicast, firewall) i opisany jest w sekcji Troubleshooting.

## Pliki w tym katalogu

| Plik | Opis |
|------|------|
| `cctv-client.service` | systemd unit (Type=simple, EnvironmentFile dla `r2.env` i `cameras.env`, ExecStart na venv `/opt/cctv-client`) |
| `install.sh` | idempotentny installer (user `cctv`, venv, deps via uv/pip, kopiowanie pakietów, etc) |
| `r2.env.example` | template dla `/etc/cctv-client/r2.env` (R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET) |
| `cameras.env.example` | template dla `/etc/cctv-client/cameras.env` (RTSP_DEFAULT_USER/PASS + opcjonalne per-IP) |

## Brak regresji dla Dockera

Docker stack (`client-agent/Dockerfile` + `docker-compose.client.yml`)
nadal działa identycznie — appliance to równoległy target, nie zamiennik.
Współdzielony pakiet `client_agent` gwarantuje 1:1 funkcjonalność: każda
nowa funkcja ląduje w obu trybach.
