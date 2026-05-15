#!/usr/bin/env bash
# Idempotent installer for the cctv-client appliance (issue #24).
#
# Run as root from inside the cloned repo:
#
#     sudo ./client-appliance/install.sh
#
# Layout it sets up:
#
#     /opt/cctv-client/            venv (python + deps + client_agent code)
#     /etc/cctv-client/r2.env       R2 creds (chmod 600, seeded from .example)
#     /etc/cctv-client/cameras.env  RTSP creds (chmod 600, seeded from .example)
#     /etc/cctv-client/platform.env GPU Exchange platform creds, optional
#                                   (chmod 600, seeded from .example, #26)
#     /etc/systemd/system/cctv-client.service
#     user `cctv`                  unprivileged owner of /opt/cctv-client
#
# Idempotent: every step guards on existing state, so a second run only
# fixes drift (re-syncing deps, re-installing the unit) and leaves operator
# edits in /etc/cctv-client/*.env untouched.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "install.sh must run as root (sudo)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "expected $REPO_ROOT/pyproject.toml — run install.sh from inside the cloned repo." >&2
    exit 1
fi

VENV=/opt/cctv-client
ETC=/etc/cctv-client
UNIT_SRC="$SCRIPT_DIR/cctv-client.service"
UNIT_DST=/etc/systemd/system/cctv-client.service

log() { printf '[install] %s\n' "$*"; }

# 1. System user — unprivileged, no shell login, home matches venv path.
if ! id -u cctv >/dev/null 2>&1; then
    log "creating system user cctv"
    useradd --system --home-dir "$VENV" --shell /usr/sbin/nologin cctv
else
    log "user cctv already exists, skipping"
fi

# 2. Venv — created once; later runs reuse it so operator edits to
#    site-packages or pinned deps survive a re-run.
if [[ ! -d "$VENV/bin" ]]; then
    log "creating venv at $VENV"
    python3 -m venv "$VENV"
else
    log "venv at $VENV already present, skipping create"
fi

# 3. Dependencies — prefer uv (matches Dockerfile path) and fall back to pip
#    so the script works on a fresh box that has not opted into uv yet.
log "installing python dependencies"
if command -v uv >/dev/null 2>&1; then
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync \
        --project "$REPO_ROOT" \
        --frozen \
        --no-dev
else
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install \
        "flask>=3.0" \
        "waitress>=3.0" \
        "boto3>=1.34" \
        "jinja2>=3.1" \
        "pillow>=10.0" \
        "numpy>=1.26,<2.0" \
        "opencv-python-headless>=4.8" \
        "WSDiscovery>=2.0" \
        "onvif-zeep>=0.2"
fi

# 4. Source modules — mirror the Dockerfile pattern (drop client_agent and
#    gpu_service into the venv site-packages so plain
#    `python -m client_agent.appliance` finds them without setting
#    PYTHONPATH in the unit).
SITE_PACKAGES="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
log "installing client_agent + gpu_service into $SITE_PACKAGES"
rm -rf "$SITE_PACKAGES/client_agent" "$SITE_PACKAGES/gpu_service"
cp -R "$REPO_ROOT/client-agent/client_agent" "$SITE_PACKAGES/client_agent"
cp -R "$REPO_ROOT/gpu-service/gpu_service" "$SITE_PACKAGES/gpu_service"

# 5. Ownership — root owns config, cctv owns the venv (so logs / cache
#    writes by the runtime user succeed).
chown -R cctv:cctv "$VENV"

# 6. /etc/cctv-client — directory 0700 root:root keeps env files (with R2
#    secrets) readable only by root and, via 0640 below, the cctv group.
log "configuring $ETC"
install -d -o root -g cctv -m 0750 "$ETC"
chmod 0700 "$ETC"

for name in r2.env cameras.env platform.env; do
    target="$ETC/$name"
    if [[ ! -f "$target" ]]; then
        log "seeding $target from $name.example"
        install -m 0600 -o root -g cctv "$SCRIPT_DIR/$name.example" "$target"
    else
        log "$target exists, leaving operator edits untouched"
    fi
    chmod 0600 "$target"
done

# 7. systemd unit — install, reload, enable. ``enable --now`` is idempotent
#    by itself but a daemon-reload after re-installing the unit is required
#    so systemd picks up directive changes.
log "installing systemd unit"
install -m 0644 "$UNIT_SRC" "$UNIT_DST"
systemctl daemon-reload
systemctl enable --now cctv-client.service

log "done — check status with: systemctl status cctv-client"
log "logs: journalctl -u cctv-client -f"
