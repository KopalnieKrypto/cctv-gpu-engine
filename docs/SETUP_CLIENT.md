# Client Setup Guide (on-premise operator)

> **The Docker client flow was retired in [#29](https://github.com/KopalnieKrypto/cctv-gpu-engine/issues/29).**
> There is no longer a `client-agent` Docker image, `docker-compose.client.yml`, or
> `.env.client`. The client now runs **bare-metal** as a systemd appliance. The
> canonical, exhaustive runbook lives in
> **[client-appliance/README.md](../client-appliance/README.md)** — install, update,
> troubleshooting, and a 5-minute smoke test. This page is just the quick redirect.

The on-premise box records video from your IP cameras and (in platform mode)
uploads it to R2 via presigned URLs. No GPU, no Docker — pure CPU. It holds **no
R2 credentials on disk**.

## Quick start

Run on a mini-PC / small Linux box (Ubuntu 24.04 LTS or Raspberry Pi OS Bookworm —
Python 3.12 + systemd + ffmpeg):

```bash
# 1. Prerequisites
sudo apt-get update && sudo apt-get install -y git python3.12-venv ffmpeg

# 2. Clone + install (idempotent — creates the cctv user, venv, and systemd unit)
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git /opt/src/cctv-gpu-engine
cd /opt/src/cctv-gpu-engine
sudo ./client-appliance/install.sh

# 3. Configure
sudo nano /etc/cctv-client/cameras.env    # RTSP_DEFAULT_USER/PASS (+ optional per-IP overrides)
sudo nano /etc/cctv-client/platform.env   # PLATFORM_URL + APPLIANCE_TOKEN for platform mode

# 4. Start on boot
sudo systemctl enable --now cctv-client
```

The Flask UI comes up on `http://<appliance-ip>:8080` — camera discovery, per-camera
snapshots, the managed-cameras panel, and the test-connection / stop controls. The
legacy manual "upload an MP4 / record → R2" routes now return **503** (retired in #29).

## Logs and status

```bash
systemctl status cctv-client
journalctl -u cctv-client -f
```

## Everything else

Camera-credential resolution (`RTSP_DEFAULT_*` + per-IP overrides), vendor RTSP URL
patterns, the ONVIF-discovery troubleshooting matrix, updating (`git pull` +
re-run `install.sh`), and the upgrade notes for existing deployments all live in
the canonical runbook:

- **[client-appliance/README.md](../client-appliance/README.md)** — client appliance install / update / troubleshooting

See also:

- [docs/SETUP_GPU.md](SETUP_GPU.md) — the GPU side that runs inference
- [README.md](../README.md) — project overview, architecture diagram
- [SPEC.md](../SPEC.md) — full technical specification
