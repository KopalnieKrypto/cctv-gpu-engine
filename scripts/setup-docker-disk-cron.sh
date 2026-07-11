#!/usr/bin/env bash
#
# setup-docker-disk-cron.sh — install a weekly Docker disk-cleanup job on a GPU server.
#
# Reclaims the two things that silently fill a GPU box's disk (see cctv-gpu-engine#19):
#   1. stopped containers + dangling images + dangling build cache  (docker system prune -f)
#   2. build cache older than 7 days                                (docker builder prune -f --filter until=168h)
#
# Deliberately SAFE / non-destructive:
#   - never touches running containers or the images they use
#   - never passes --volumes  (named volumes can hold data — prune those by hand)
#   - never passes -a to system prune  (tagged images like :latest survive → no forced re-pull)
#
# Portable across GPU hosts:
#   - root / passwordless-sudo  → system-wide /etc/cron.weekly/docker-prune  (runs via anacron)
#   - no sudo (e.g. cctv-vps-2) → per-user crontab entry, Sundays 04:00
# Idempotent: safe to re-run; re-running refreshes the managed entry in place.
#
# Usage:  bash scripts/setup-docker-disk-cron.sh          # install
#         bash scripts/setup-docker-disk-cron.sh --dry-run # show what would run, then run one prune now
#
set -euo pipefail

MARKER='cctv-docker-prune (managed by scripts/setup-docker-disk-cron.sh)'
PRUNE_CMD='docker system prune -f && docker builder prune -f --filter until=168h'

log() { printf '[docker-disk-cron] %s\n' "$*"; }

# --- 1. make sure we can actually reach a docker daemon, remember which socket works ---
if docker version >/dev/null 2>&1; then
  RESOLVED_HOST="${DOCKER_HOST:-}"   # default context works; keep whatever (maybe empty) is set
else
  export DOCKER_HOST="unix:///var/run/docker.sock"   # rootless is often broken; fall back to system socket
  if docker version >/dev/null 2>&1; then
    RESOLVED_HOST="$DOCKER_HOST"
    log "note: default docker context failed; using $DOCKER_HOST"
  else
    log "ERROR: cannot reach a docker daemon (tried default context and unix:///var/run/docker.sock)."
    log "       add your user to the 'docker' group, or export a working DOCKER_HOST, then re-run."
    exit 1
  fi
fi

# The exact command the cron will run — bakes in the working socket so cron matches interactive behaviour.
if [ -n "$RESOLVED_HOST" ]; then
  CRON_CMD="export DOCKER_HOST=$RESOLVED_HOST; $PRUNE_CMD"
else
  CRON_CMD="$PRUNE_CMD"
fi

if [ "${1:-}" = "--dry-run" ]; then
  log "would install cron command:"
  printf '    %s\n' "$CRON_CMD"
  log "running one prune now (dry-run mode):"
  bash -c "$CRON_CMD"
  df -h / | tail -1
  exit 0
fi

install_system() {   # $1 = optional sudo prefix
  local sudo="$1" script=/etc/cron.weekly/docker-prune
  $sudo tee "$script" >/dev/null <<EOF
#!/bin/sh
# $MARKER
$CRON_CMD
EOF
  $sudo chmod 0755 "$script"
  log "installed system job: $script (runs weekly via cron.weekly/anacron, as root)"
}

install_user() {
  local tmp; tmp="$(mktemp)"
  # drop any previous managed entry (dedupe), keep everything else, append a fresh one
  { crontab -l 2>/dev/null || true; } | grep -vF "$MARKER" > "$tmp"
  printf '0 4 * * 0 %s # %s\n' "$CRON_CMD" "$MARKER" >> "$tmp"
  crontab "$tmp"
  rm -f "$tmp"
  log "installed user crontab entry for $(id -un): Sundays 04:00"
}

if [ "$(id -u)" -eq 0 ]; then
  install_system ""
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  install_system "sudo"
else
  log "no root / passwordless sudo — falling back to a per-user crontab entry"
  install_user
fi

log "current disk usage:"; df -h / | tail -1
log "done. To verify: sudo cat /etc/cron.weekly/docker-prune   (system)   or   crontab -l   (per-user)"
