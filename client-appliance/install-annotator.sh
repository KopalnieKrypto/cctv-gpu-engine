#!/usr/bin/env bash
# Installs zone-annotator behind the panel's proxy, and the box's tunnel (gpu-exchange#298).
#
# Run as root on the box, after copying the package next to this checkout:
#
#     sudo ./client-appliance/install-annotator.sh <agent-user> <zone-annotator.tar> <appliance uuid>
#
# The appliance uuid is the one the panel shows; the box itself keeps no record of it, and it
# names this box's key on cctv-vps (gpu-exchange#300).
#
# The package is `git archive` of the commit in zone-annotator.pin, made on the operator's
# machine because the zone-annotator repo is private:
#
#     git -C ../zone-annotator archive -o zone-annotator.tar "$(cat client-appliance/zone-annotator.pin)"
#
# Layout:
#
#     /opt/zone-annotator/releases/<commit>/   the tool and its venv, root-owned
#     /opt/zone-annotator/current              symlink to the release the unit runs
#
# DESTDIR prefixes every root-owned path and lifts the root check, so the tests run this script
# for real; SYSTEMCTL and RUNUSER stand in for systemctl and runuser there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESTDIR="${DESTDIR:-}"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"
RUNUSER="${RUNUSER:-runuser}"

log() { printf '[install-annotator] %s\n' "$*"; }
die() { printf '[install-annotator] %s\n' "$*" >&2; exit 1; }

[[ $# -eq 3 ]] || die "usage: $0 <agent-user> <zone-annotator.tar> <appliance uuid>"
AGENT="$1"
PACKAGE="$2"
APPLIANCE="$3"
[[ $APPLIANCE =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    die "appliance id must be the appliance's lowercase UUID, as the panel shows it"

if [[ -z "$DESTDIR" && $EUID -ne 0 ]]; then
    die "must run as root (sudo)"
fi

# cloudflared reaches the edge on 7844 and nothing else will do, so check before touching the box.
# TCP only: a UDP probe cannot tell a drop from silence, and cloudflared falls back to TCP on its own.
EGRESS_HOST="${EGRESS_HOST:-region1.v2.argotunnel.com}"
EGRESS_PORT="${EGRESS_PORT:-7844}"
timeout 5 bash -c "exec 3<>/dev/tcp/$EGRESS_HOST/$EGRESS_PORT" 2>/dev/null ||
    die "no outbound TCP to $EGRESS_HOST:$EGRESS_PORT: the box's tunnel needs port 7844 open to Cloudflare"

# Packages leave for cctv-vps over rsync (gpu-exchange#300); without it nothing here can ship.
command -v "${RSYNC:-rsync}" >/dev/null || die "rsync not installed: sudo apt-get install rsync"
# sudo runs this with secure_path, which rarely holds the ~/.local/bin uv usually lives in; the
# release is replaced before uv runs, so a missing uv would leave the box with none.
UV="${UV:-uv}"
command -v "$UV" >/dev/null ||
    die "uv not on root's PATH under sudo (secure_path): install it where root finds it, e.g. /usr/local/bin"
GIT="${GIT:-git}"
command -v "$GIT" >/dev/null || die "git not installed: sudo apt-get install git"

# The agent owns its home and may leave links in it, so root changes nothing there itself: as
# root, ssh-keygen or chown would follow a link anywhere.
as_agent() { "$RUNUSER" -u "$AGENT" -- "$@"; }

PIN="${ZONE_ANNOTATOR_PIN:-$(tr -d '[:space:]' <"$SCRIPT_DIR/zone-annotator.pin")}"
PREFIX="$DESTDIR/opt/zone-annotator"
RELEASE="$PREFIX/releases/$PIN"

# The package usually sits in the agent's home, where the agent can swap it between a check
# and the unpacking. So it is read once, as the agent, into a directory only root can write,
# and that copy is what is checked and unpacked.
WORK="$(mktemp -d)"
SMOKE=""
cleanup() { rm -rf "$WORK"; [[ -z "$SMOKE" ]] || as_agent rm -rf "$SMOKE"; }
trap cleanup EXIT
TAR="$WORK/zone-annotator.tar"
as_agent cat "$PACKAGE" >"$TAR" || die "$AGENT cannot read $PACKAGE: copy the package into its home"

# The commit git wrote into the archive itself; a file name can be renamed, this cannot.
PACKAGE_COMMIT="$("$GIT" get-tar-commit-id <"$TAR" || true)"
[[ "$PACKAGE_COMMIT" == "$PIN" ]] ||
    die "package is ${PACKAGE_COMMIT:0:7}, zone-annotator.pin wants ${PIN:0:7}; cut it again with git archive"

CHANGED=0
if [[ "$(readlink "$PREFIX/current" 2>/dev/null)" == "releases/$PIN" && -x "$RELEASE/.venv/bin/python" ]]; then
    log "zone-annotator ${PIN:0:7} already installed"
else
    log "installing zone-annotator ${PIN:0:7}"
    rm -rf "$RELEASE"
    mkdir -p "$RELEASE"
    tar -xf "$TAR" -C "$RELEASE"
    # System python, never a uv-managed one: uv would download it into root's home, which the
    # service user cannot read.
    UV_PROJECT_ENVIRONMENT="$RELEASE/.venv" UV_PYTHON_DOWNLOADS=never \
        "$UV" sync --project "$RELEASE" --frozen --no-dev --python python3 -q
    # Code from the package runs as the agent, the user the service runs as anyway.
    SMOKE="$(as_agent mktemp -d)"
    (cd "$RELEASE" && as_agent .venv/bin/python -m zone_annotator.smoke_phase1 --out "$SMOKE/out")
    ln -sfn "releases/$PIN" "$PREFIX/current"
    find "$PREFIX/releases" -mindepth 1 -maxdepth 1 ! -name "$PIN" -exec rm -rf {} +
    CHANGED=1
fi

# Footage the agent cuts for the panel, and the projects the tool writes.
AGENT_HOME="${AGENT_HOME:-$(getent passwd "$AGENT" | cut -d: -f6)}"
GROUP="$(id -gn "$AGENT")"
STATE="$AGENT_HOME/.local/state/zone-annotator"
as_agent install -d -m 0750 "$STATE/windows" "$STATE/projects"

# Writes a unit only when its text differs and says so in UNIT_CHANGED, not in its status: a
# caller testing that status would switch set -e off inside.
put_unit() {
    local path="$DESTDIR/etc/systemd/system/$1"
    UNIT_CHANGED=0
    [[ -f "$path" && "$(cat "$path")" == "$2" ]] && return
    log "installing $path"
    mkdir -p "$(dirname "$path")"
    printf '%s\n' "$2" >"$path"
    "$SYSTEMCTL" daemon-reload
    UNIT_CHANGED=1
}

# A re-run is also the repair after a failed audit, so it brings back a stopped or disabled unit.
run_unit() {
    if (($2)); then
        "$SYSTEMCTL" enable "$1"
        "$SYSTEMCTL" restart "$1"
    else
        "$SYSTEMCTL" is-enabled --quiet "$1" || "$SYSTEMCTL" enable "$1"
        "$SYSTEMCTL" is-active --quiet "$1" || "$SYSTEMCTL" start "$1"
    fi
}

put_unit zone-annotator.service "$(sed -e "s|@AGENT@|$AGENT|g" -e "s|@GROUP@|$GROUP|g" \
    -e "s|@HOME@|$AGENT_HOME|g" -e "s|@PROJECTS@|$STATE/projects|g" -e "s|@WINDOWS@|$STATE/windows|g" \
    "$SCRIPT_DIR/zone-annotator.service")"
CHANGED=$((CHANGED | UNIT_CHANGED))
run_unit zone-annotator.service "$CHANGED"

# cloudflared, pinned and checksummed as in gpu-exchange's install-cloudflared-sglang.sh: bump the
# version and both hashes together (github.com/cloudflare/cloudflared/releases).
CLOUDFLARED="${CLOUDFLARED:-/usr/bin/cloudflared}"
CLOUDFLARED_VERSION=2026.7.1
if [[ "$("$CLOUDFLARED" --version 2>/dev/null)" == *"version $CLOUDFLARED_VERSION "* ]]; then
    log "cloudflared $CLOUDFLARED_VERSION already installed"
else
    ARCH="$(dpkg --print-architecture)"
    case "$ARCH" in
        amd64) CLOUDFLARED_SHA256=79f790b45e6a9152c6cf63f60f4901e3d8a029f7f4be1345a24cd2373aba8e7d ;;
        arm64) CLOUDFLARED_SHA256=4cc3570b4f86ca8b90d7d86d327a2138ade04905d5177a98652f461292725043 ;;
        *) die "no pinned cloudflared checksum for $ARCH" ;;
    esac
    log "installing cloudflared $CLOUDFLARED_VERSION"
    DEB="$(mktemp --suffix=.deb)"
    curl -fsSL -o "$DEB" \
        "https://github.com/cloudflare/cloudflared/releases/download/$CLOUDFLARED_VERSION/cloudflared-linux-$ARCH.deb"
    echo "$CLOUDFLARED_SHA256  $DEB" | sha256sum -c - >/dev/null ||
        { rm -f "$DEB"; die "cloudflared $CLOUDFLARED_VERSION ($ARCH) checksum mismatch, refusing to install"; }
    dpkg -i "$DEB" >/dev/null
    rm -f "$DEB"
fi

# The tunnel script in gpu-exchange stages the connector token over ssh in the agent's config,
# which zone-annotator cannot read; from here only root and the tunnel's credential can.
TOKEN="$DESTDIR/etc/zone-annotator/tunnel.token"
STAGED="$AGENT_HOME/.config/cctv-client/zone-annotator-tunnel.token"
TUNNEL_CHANGED=0
if [[ -f "$STAGED" ]]; then
    log "installing the tunnel token"
    # 0755, not 0700: the audit stats the token as the unprivileged user; the file stays 0600.
    install -d -m 0755 "$(dirname "$TOKEN")"
    as_agent cat "$STAGED" | install -m 0600 /dev/stdin "$TOKEN"
    as_agent rm -f "$STAGED"
    TUNNEL_CHANGED=1
elif [[ -f "$TOKEN" ]]; then
    chmod 0755 "$(dirname "$TOKEN")"
    chmod 0600 "$TOKEN"
fi
put_unit zone-annotator-tunnel.service "$(cat "$SCRIPT_DIR/zone-annotator-tunnel.service")"
TUNNEL_CHANGED=$((TUNNEL_CHANGED | UNIT_CHANGED))
if [[ -f "$TOKEN" ]]; then
    run_unit zone-annotator-tunnel.service "$TUNNEL_CHANGED"
else
    log "no tunnel token yet: run gpu-exchange's appliance tunnel script for this box, then this installer again"
fi

# The key packages leave with, over rsync into cctv-vps's drop account. In ~/.ssh, which the
# zone-annotator unit cannot see (InaccessiblePaths); a re-run keeps the key and repairs its mode.
DROP_KEY="$AGENT_HOME/.ssh/annotation-drop"
[[ -d "$AGENT_HOME/.ssh" ]] || as_agent install -d -m 0700 "$AGENT_HOME/.ssh"
if [[ ! -f "$DROP_KEY" ]]; then
    log "creating the shipping key $DROP_KEY"
    as_agent ssh-keygen -q -t ed25519 -N "" -C "appliance-$APPLIANCE" -f "$DROP_KEY"
fi
as_agent chmod 0600 "$DROP_KEY"
# cctv-vps's host key as pinned in the repo: the poller trusts nothing else (StrictHostKeyChecking=yes),
# so a key learned on first use, possibly from the client's LAN, is replaced.
KNOWN="$DROP_KEY.known_hosts"
if ! as_agent cmp -s - "$KNOWN" <"$SCRIPT_DIR/annotation-drop.known_hosts"; then
    log "installing cctv-vps's pinned host key $KNOWN"
    as_agent install -m 0644 /dev/stdin "$KNOWN" <"$SCRIPT_DIR/annotation-drop.known_hosts"
fi
read -r _type DROP_PUB _comment < <(as_agent cat "$DROP_KEY.pub")
log "register this box's shipping key on cctv-vps (gpu-exchange scripts/provision):"
log "  sudo scripts/provision/annotation-drop.sh add $APPLIANCE \"ssh-ed25519 $DROP_PUB\""

log "done. The agent cuts windows only with both of these in its platform.env:"
log "  ANNOTATION_WINDOWS_DIR=$STATE/windows"
log "  ANNOTATION_PROJECTS_DIR=$STATE/projects"
