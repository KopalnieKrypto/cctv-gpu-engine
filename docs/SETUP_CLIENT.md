# Client Setup Guide

The Docker client flow was retired in [#29](https://github.com/KopalnieKrypto/cctv-gpu-engine/issues/29). The client now runs only as the bare-metal `client_agent.appliance` systemd service and never stores R2 credentials.

The canonical exhaustive runbook is [client-appliance/README.md](../client-appliance/README.md). This page is the short installation reference.

## Requirements

- Ubuntu 24.04 LTS or Raspberry Pi OS Bookworm;
- Python 3.12, systemd, ffmpeg, and git;
- network access to the RTSP cameras;
- outbound HTTPS to GPU Exchange in platform mode;
- no GPU and no Docker.

## Choose an installation mode

### Root/system install

Use this when sudo is available.

```bash
sudo apt-get update
sudo apt-get install -y git python3.12-venv ffmpeg

git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git /opt/src/cctv-gpu-engine
cd /opt/src/cctv-gpu-engine
sudo ./client-appliance/install.sh

sudo nano /etc/cctv-client/cameras.env
sudo nano /etc/cctv-client/platform.env
sudo systemctl enable --now cctv-client
```

Status and logs:

```bash
systemctl status cctv-client
journalctl -u cctv-client -f
```

### User-mode install

Use this when sudo is unavailable.

```bash
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git ~/cctv-gpu-engine
cd ~/cctv-gpu-engine
./client-appliance/install-user.sh

nano ~/.config/cctv-client/cameras.env
nano ~/.config/cctv-client/platform.env
systemctl --user status cctv-client
```

Status and logs:

```bash
systemctl --user status cctv-client
journalctl --user -u cctv-client -f
```

`install-user.sh` enables and verifies systemd linger so the service survives logout and reboot. If site policy blocks self-service linger, an administrator must run `sudo loginctl enable-linger <user>` once.

## Operating modes

| Mode | Configuration | Behavior |
|---|---|---|
| Standalone | Camera env only | UI, discovery, snapshots, and buffer-only recorder; no platform callbacks |
| Platform | Both `PLATFORM_URL` and `APPLIANCE_TOKEN` | Registration, heartbeat configuration, managed recorders, rolling retention, task/snapshot polling, presigned uploads |

The UI listens on `http://<appliance-ip>:8080`.

Legacy manual `/upload`, `/start`, `/jobs`, and `/report` R2 workflows return `503`. The client has no direct R2 client or credentials.

## Platform runtime settings

Environment values are cold-start fallbacks:

| Environment | Default | Platform key |
|---|---:|---|
| `BUFFER_HOURS` | `1` | `buffer_hours` |
| `POLLING_INTERVAL_SECONDS` | `5` | `polling_interval_seconds` |
| `HEARTBEAT_INTERVAL_SECONDS` | `30` | `heartbeat_interval_seconds` |
| `UPLOAD_CHUNK_BYTES` | `52428800` | `upload_chunk_bytes` |

All values must be positive integers. Valid settings returned by register/heartbeat override environment fallbacks and apply live without restarting the appliance. Invalid platform values are ignored with a warning while the last valid value remains active.

See `client-appliance/platform.env.example` and the canonical runbook for full precedence and troubleshooting.

## Updating

Root install:

```bash
cd /opt/src/cctv-gpu-engine
sudo git pull --ff-only
sudo ./client-appliance/install.sh
sudo systemctl restart cctv-client
```

User install:

```bash
cd ~/cctv-gpu-engine
git pull --ff-only
./client-appliance/install-user.sh
systemctl --user restart cctv-client
```

Both installers preserve the operator-edited environment files.

## Smoke procedure

Install on the representative target, start the service, discover a camera, open a snapshot, perform a short recording, and confirm a chunk appears in the local rolling buffer. In platform mode, also verify register/heartbeat logs and one presigned upload.

Do not quote a generic installation ETA without measuring this procedure on representative hardware and network conditions.

## See also

- [canonical appliance runbook](../client-appliance/README.md)
- [GPU setup](SETUP_GPU.md)
- [project overview](../README.md)
- [current specification](../SPEC.md)
