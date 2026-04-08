# Client Agent — Setup Guide (on-premise operator)

Step-by-step instructions to bring up `cctv-client-agent` on a regular CPU
machine inside the customer LAN. This is the **on-premise side** of the
system: it records video from your IP cameras (or accepts manual MP4 uploads),
ships the footage to Cloudflare R2, and shows you the analysis report when
the GPU side is done.

> **TL;DR for the experienced operator:**
> any Docker host → clone repo → `cp .env.client.example .env.client` and fill
> R2 keys → `docker compose -f docker-compose.client.yml up -d` → open
> `http://localhost:8080`.

---

## 1. Hardware requirements

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| CPU | 2 cores x86_64 or arm64 | 4 cores | The agent does **no inference** — it only does ffmpeg stream-copy (`-c copy`, no re-encoding) and S3 multipart uploads. Even a Raspberry Pi 4 is enough. |
| RAM | 1 GB free | 2 GB | ffmpeg stream-copy is constant-memory; no full video is buffered. |
| Disk | 5 GB free | 50 GB free | Workdir for in-flight chunks before they upload to R2. Sized for the longest recording you intend to run × your camera bitrate (e.g. 8 h × 4 Mbps ≈ 14 GB). |
| Network — to cameras | LAN access to the camera RTSP URL (typically port 554) | Wired Ethernet | RTSP over TCP works through most NATs; the agent does not need to be on the same subnet as the cameras as long as routing is in place. |
| Network — to R2 | Outbound HTTPS to `*.r2.cloudflarestorage.com` | ≥ 5 Mbps upload | Sustained upload bandwidth becomes the bottleneck for long recordings. |
| GPU | **None** | **None** | The agent is pure CPU. Do not waste a GPU host on this role. |

The agent does **not** need a public IP, port forwarding, or a VPN. All
communication with the GPU side is one-way outbound to R2.

## 2. Software requirements (one-time host setup)

### 2.1 Operating system

Anything Docker runs on:

- Linux x86_64 / arm64 (Ubuntu 22.04+, Debian 12+, Raspberry Pi OS, etc.)
- Windows 10/11 with **Docker Desktop**
- macOS with **Docker Desktop**

Pick whatever the customer site already has — there is no platform-specific
optimization in the agent.

### 2.2 Docker Engine ≥ 24.0 with the `compose` plugin

**Linux:**

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER         # log out / log back in afterwards
docker --version                      # 24.0+
docker compose version                # v2.x
```

**Windows / macOS:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
(latest stable). It bundles both the engine and the compose plugin.

### 2.3 Git

Used only to clone the repository (for the compose file and `.env.client.example`).
The container image is pulled from GHCR.

### 2.4 No ffmpeg installation needed

ffmpeg is bundled **inside** the container image. You do not install it on the
host.

## 3. Cloudflare R2 credentials

The R2 bucket is shared with the GPU host operator. **They typically create
the bucket and the API token, then send the values to you out-of-band**
(encrypted email, password manager share, etc.).

You need four values:

| Variable | Example | Where it comes from |
|---|---|---|
| `R2_ENDPOINT` | `https://abc123def.r2.cloudflarestorage.com` | Bucket overview page in the Cloudflare dashboard |
| `R2_ACCESS_KEY_ID` | (32 hex chars) | "Manage R2 API Tokens" → token detail |
| `R2_SECRET_ACCESS_KEY` | (64 hex chars) | Same — shown only at token creation |
| `R2_BUCKET` | `surveillance-data` | The bucket name itself; must match the GPU side exactly |

If you also operate the GPU host, see [docs/SETUP_GPU.md §3](SETUP_GPU.md#3-cloudflare-r2-credentials)
for how to create the bucket and token.

## 4. Bring the agent up

### 4.1 Clone the repository

```bash
git clone https://github.com/KopalnieKrypto/cctv-gpu-engine.git
cd cctv-gpu-engine
```

You only need the repo for the `docker-compose.client.yml` file and the
`.env.client.example` template. The image itself comes from GHCR.

### 4.2 Configure environment

```bash
cp .env.client.example .env.client
$EDITOR .env.client
```

Fill in:

```dotenv
# R2 credentials (from §3 — same bucket as the GPU side)
R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<from operator>
R2_SECRET_ACCESS_KEY=<from operator>
R2_BUCKET=surveillance-data

# Default RTSP URL prefilled in the recording form (optional —
# you can also paste it in the UI each time).
RTSP_DEFAULT_URL=rtsp://camera-user:camera-pass@192.168.1.50:554/Streaming/Channels/101

# Hard cap on a single recording session — protects the workdir disk
# from runaway sessions. The UI only allows {1, 2, 4, 8} hours.
MAX_RECORDING_HOURS=8
```

The RTSP URL format depends on your camera vendor. Common patterns:

| Vendor | URL pattern |
|---|---|
| Hikvision | `rtsp://user:pass@host:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@host:554/cam/realmonitor?channel=1&subtype=0` |
| Axis | `rtsp://user:pass@host:554/axis-media/media.amp` |
| Reolink | `rtsp://user:pass@host:554/h264Preview_01_main` |

When in doubt, check the camera's web UI under "ONVIF" or "RTSP" settings, or
test the URL with VLC (Media → Open Network Stream).

### 4.3 Pull and start

```bash
docker compose -f docker-compose.client.yml pull
docker compose -f docker-compose.client.yml up -d
```

`pull` grabs `ghcr.io/kopalniekrypto/cctv-gpu-engine/client-agent:latest` from
the public GHCR registry. No login required.

## 5. Verify the deployment

```bash
docker compose -f docker-compose.client.yml ps           # cctv-client-agent "Up"
docker compose -f docker-compose.client.yml logs -f      # Flask app boot logs
```

Open `http://localhost:8080` in a browser (or `http://<agent-host-ip>:8080`
from another machine on the LAN). You should see:

- **Upload an MP4** form — for manual analysis of pre-recorded footage
- **Record from RTSP camera** form — test connection / start / stop controls
- **View job list** link — table of all jobs with status badges that auto-refresh
- (If a recording is in progress) a banner showing the current state

### 5.1 Smoke test the camera connection

In the **Test connection** field, paste your RTSP URL and click *Test
connection*. The agent will:

1. Spawn `ffmpeg -rtsp_transport tcp -i <url> -t 2 -f null -` for two seconds
2. Return `OK` if ffmpeg sees video frames, or the error message verbatim
   otherwise

This is the fastest way to validate the URL, the camera credentials, and
network reachability without committing to a multi-hour recording.

### 5.2 Smoke test an upload

Pick any short MP4 with people in it (a phone clip works), upload it via the
form, then watch the **View job list** page. The job will cycle through:

`pending` → `processing` → `done` (or `failed`)

`pending` → `processing` requires the GPU side to be online and polling — if
it stays `pending` for more than ~30 seconds, the GPU side isn't picking up
jobs. See [docs/SETUP_GPU.md §7](SETUP_GPU.md#7-troubleshooting).

When it reaches `done`, the **view** link opens the standalone HTML report;
**download** saves it to disk.

## 6. Recording from a camera

1. Open `http://localhost:8080`
2. Paste the RTSP URL into the recording form
3. Pick a duration: **1 / 2 / 4 / 8 hours** (capped by `MAX_RECORDING_HOURS`)
4. Click **Start recording**

The agent will:

1. Spawn ffmpeg with `-c copy` (no re-encoding, near-zero CPU) into rolling
   chunks in the workdir
2. Upload each chunk to R2 as it closes (multipart upload, never the whole
   file at once)
3. Mark the job as `pending` so the GPU side picks it up

You can leave the browser tab — recordings continue in the container, not in
the page. Click **Stop current recording** to abort early; ffmpeg gets a
clean SIGINT and the partial recording is still uploaded and queued.

Only one recording runs at a time per container. If you need parallel
recordings from multiple cameras, run multiple `cctv-client-agent` containers
on different host ports — they all upload to the same R2 bucket and the GPU
side handles the queue.

## 7. Troubleshooting

| Symptom | Diagnosis | Fix |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker not running / not in `docker` group | `sudo systemctl start docker` + `usermod -aG docker $USER` + relog. |
| `http://localhost:8080` refuses connection | Container exited at boot — bad env file | `docker compose -f docker-compose.client.yml logs` — look for `R2Client` init errors. |
| Test connection returns `403` / `401` from RTSP | Wrong camera credentials or URL path | Confirm with VLC first, then copy the working URL into the form. |
| Test connection returns `Connection refused` | Wrong IP, wrong port, or camera firewall | Check the camera is reachable: `ping <camera-ip>` and `nc -zv <camera-ip> 554`. |
| Job stays `pending` forever | GPU side is offline or pointed at a different bucket | Check [GPU §7](SETUP_GPU.md#7-troubleshooting). Confirm `R2_BUCKET` and `R2_ENDPOINT` match on both sides. |
| Job goes `failed` immediately | GPU side fetched the chunks but inference crashed | `docker compose logs gpu-service` on the **GPU** host. The error message is also written to `status.json` in R2. |
| Workdir filling up the host disk | Long recording + slow upload bandwidth means chunks queue up locally | Lower `MAX_RECORDING_HOURS`, or upgrade upload bandwidth, or wipe `cctv-client-workdir` volume after stopping the container. |

## 8. Updating

```bash
cd cctv-gpu-engine
git pull
docker compose -f docker-compose.client.yml pull
docker compose -f docker-compose.client.yml up -d
```

The `:latest` tag is rebuilt on every push to `main`. To pin a frozen version,
edit `docker-compose.client.yml` and replace `:latest` with a specific tag
(e.g. `:sha-450dccb`).

## 9. Stopping / removing

```bash
docker compose -f docker-compose.client.yml down            # stop, keep workdir
docker compose -f docker-compose.client.yml down -v         # also wipe workdir
```

In-flight chunks in the workdir are lost on `down -v`. Anything already
uploaded to R2 is **not** touched.

---

See also:

- [docs/SETUP_GPU.md](SETUP_GPU.md) — the GPU side that runs inference
- [README.md](../README.md) — project overview, architecture diagram
- [SPEC.md](../SPEC.md) — full technical specification
