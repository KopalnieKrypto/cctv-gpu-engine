#!/usr/bin/env bash
# No-root installer for the cctv-client appliance (issue #84).
#
# For sites where install.sh cannot be used because no sudo password is
# available (see gpu-exchange:docs/ops/appliance-sites.md). Run as the
# *unprivileged* user that will own the appliance, from inside the clone:
#
#     ./client-appliance/install-user.sh
#
# Layout it sets up — everything under $HOME, nothing root-owned:
#
#     ~/.local/share/cctv-client/           venv (python + deps + client_agent)
#     ~/.config/cctv-client/cameras.env     RTSP creds (chmod 600, seeded from .example)
#     ~/.config/cctv-client/platform.env    GPU Exchange creds, optional (chmod 600)
#     ~/.config/systemd/user/cctv-client.service
#
# Idempotent: every step guards on existing state, so a re-run only fixes
# drift and leaves operator edits in ~/.config/cctv-client/*.env untouched.
#
# Why this exists: before #84 a no-sudo site ran the appliance from a
# hand-rolled `nohup setsid` one-liner with zero supervision — no restart on
# crash, no start on boot. That cost a ~18 h outage on cameraboy
# (2026-07-14 → 2026-07-16), during which a queued task aged its footage out
# of the 1 h rolling buffer and could never be recovered.
set -euo pipefail

if [[ $EUID -eq 0 ]]; then
    echo "install-user.sh must NOT run as root — use install.sh for the system-wide install." >&2
    echo "Running this under sudo would leave root-owned files in \$HOME and would enable" >&2
    echo "the unit in *root's* user manager, which does not survive reboot as this user." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "expected $REPO_ROOT/pyproject.toml — run install-user.sh from inside the cloned repo." >&2
    exit 1
fi

VENV="$HOME/.local/share/cctv-client"
CONF="$HOME/.config/cctv-client"
UNIT_SRC="$SCRIPT_DIR/cctv-client-user.service"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT_DST="$UNIT_DIR/cctv-client.service"
ME="$(id -un)"

log() { printf '[install-user] %s\n' "$*"; }

# 1. Venv — created once; later runs reuse it.
if [[ ! -d "$VENV/bin" ]]; then
    log "creating venv at $VENV"
    python3 -m venv "$VENV"
else
    log "venv at $VENV already present, skipping create"
fi

# 2. Dependencies — prefer uv (matches install.sh / Dockerfile), fall back to
#    pip so a box that never opted into uv still installs.
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
        "onvif-zeep>=0.2" \
        "tinytuya>=1.13"
fi

# 3. Source modules — drop client_agent into the venv site-packages so the
#    unit can run plain `python -m client_agent.appliance` with no PYTHONPATH.
#    The pre-#84 hand-rolled launcher on cameraboy depended on an exported
#    PYTHONPATH=<repo>/client-agent; a systemd unit inherits no such shell
#    environment, so that dependency has to be designed out rather than
#    re-encoded as an Environment= line.
SITE_PACKAGES="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
log "installing client_agent into $SITE_PACKAGES"
rm -rf "$SITE_PACKAGES/client_agent"
cp -R "$REPO_ROOT/client-agent/client_agent" "$SITE_PACKAGES/client_agent"

# 4. Config dir — 0700 keeps the env files (RTSP + platform secrets) readable
#    only by this user. No group needed: single-user layout.
log "configuring $CONF"
mkdir -p "$CONF"
chmod 0700 "$CONF"

for name in cameras.env platform.env; do
    target="$CONF/$name"
    if [[ ! -f "$target" ]]; then
        log "seeding $target from $name.example"
        cp "$SCRIPT_DIR/$name.example" "$target"
    else
        log "$target exists, leaving operator edits untouched"
    fi
    chmod 0600 "$target"
done

# 5. Unit file — shipped as cctv-client-user.service, installed under the
#    canonical name so operator muscle memory (`systemctl --user status
#    cctv-client`) matches the root install.
log "installing user unit to $UNIT_DST"
mkdir -p "$UNIT_DIR"
install -m 0644 "$UNIT_SRC" "$UNIT_DST"

# 6. Linger — the load-bearing step. Without it the user manager (and this
#    unit with it) is torn down when the last session for this user ends, so
#    the appliance would survive `exit` but die on reboot: the same silent
#    failure #84 exists to kill, just deferred.
#
#    `loginctl enable-linger <self>` maps to the polkit action
#    org.freedesktop.login1.set-self-linger, which ships allow_any=yes on
#    Ubuntu 24.04 / systemd 255 — no root, no password, no polkit prompt.
#    Naming a *different* user maps to set-user-linger, which is auth-gated.
if [[ "$(loginctl show-user "$ME" --property=Linger --value 2>/dev/null || echo no)" == "yes" ]]; then
    log "linger already enabled for $ME, skipping"
else
    log "enabling linger for $ME (so the unit starts at boot with no SSH session)"
    loginctl enable-linger "$ME" || true
fi

# 7. Verify linger actually took. Never assume: on a site whose polkit policy
#    differs from the probed default this is where it fails, and a silent
#    failure here means an appliance that looks healthy until the next reboot
#    and then is gone — the failure mode is missing footage hours later, not
#    a red install now.
if [[ "$(loginctl show-user "$ME" --property=Linger --value 2>/dev/null || echo no)" != "yes" ]]; then
    echo "ERROR: linger is still disabled for $ME." >&2
    echo "The appliance will run now but will NOT come back after a reboot." >&2
    echo "Ask an admin for a one-time: sudo loginctl enable-linger $ME" >&2
    echo "then re-run this script. See client-appliance/README.md (tryb user-mode)." >&2
    exit 1
fi
log "linger confirmed enabled for $ME"

# 8. Enable + start. `enable` wires into default.target (the user manager's
#    boot target — it has no multi-user.target); `--now` starts it this boot.
#    daemon-reload first so a re-run picks up changed unit directives.
#
#    Capture "was it already running" BEFORE enable --now, because after it the
#    answer is always yes and the distinction is lost.
WAS_ACTIVE=0
if systemctl --user is-active --quiet cctv-client.service 2>/dev/null; then
    WAS_ACTIVE=1
fi

log "enabling and starting the unit"
systemctl --user daemon-reload
systemctl --user enable --now cctv-client.service

# `enable --now` starts a *stopped* unit but no-ops on a running one. On an
# update that means step 3 copied new sources into site-packages while the
# live process keeps executing the code it imported at ITS start — Python
# loads at import, so replacing files under a running process changes nothing
# about what actually runs. The box then reports one version on disk and
# executes another in memory, which is exactly the state that hid a stale
# build on cameraboy for three days (2026-07-20).
#
# So: restart only when the unit was already up. A unit that was *stopped* has
# just been started by --now with the new code and must not be bounced again.
if [[ "$WAS_ACTIVE" -eq 1 ]]; then
    log "unit was already running — restarting so it imports the newly installed code"
    systemctl --user restart cctv-client.service
fi

log "done — check status with: systemctl --user status cctv-client"
log "logs: journalctl --user -u cctv-client -f"
