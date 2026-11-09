#!/usr/bin/env bash
# harden-appliance.sh - close a new cameraboy to the client LAN. Run once, as root.
#
# Usage: sudo client-appliance/harden-appliance.sh <appliance-user>
#
# The panel binds 0.0.0.0:8080 with no login (live camera snapshots, discovery,
# RTSP test), so only the host firewall keeps the client's LAN off it. This is
# the root half that install-user.sh cannot do:
#
#   ufw   deny incoming except tailscale0, Tailscale's direct-path UDP 41641 and
#         the LAN replies discovery needs (ONVIF from port 3702, Tuya to 6666-6667)
#   sshd  PasswordAuthentication no in a drop-in that sorts before cloud-init's
#   linger for <appliance-user>, so the user unit starts at boot
#
# Both preconditions are checked before anything changes: without tailscale0 the
# deny policy cuts the only remote way in, and key-only SSH without a key locks
# everyone out. Idempotent: ufw dedupes rules, the drop-in is rewritten as-is.
# audit-appliance.sh checks 5 and 6 verify the result without sudo.
#
# Source incident (2026-11-09): cameraboy had ufw inactive and sshd taking
# passwords; cameraboy2 had been closed by a script that never reached the repo.
set -euo pipefail

SSHD_DROPIN_DIR="${SSHD_DROPIN_DIR:-/etc/ssh/sshd_config.d}"

user="${1:-}"
if [[ -z "$user" ]]; then
  echo "usage: $0 <appliance-user>" >&2
  exit 64
fi
if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root: sudo $0 $user" >&2
  exit 1
fi
home="$(getent passwd "$user" | cut -d: -f6)"
if [[ -z "$home" ]]; then
  echo "no such user: $user" >&2
  exit 1
fi
if ! ip link show tailscale0 >/dev/null 2>&1; then
  echo "tailscale0 not found - bring Tailscale up first; deny-incoming now would cut remote SSH" >&2
  exit 1
fi
if [[ ! -s "$home/.ssh/authorized_keys" ]]; then
  echo "$home/.ssh/authorized_keys is empty - add a key first; key-only SSH now would lock everyone out" >&2
  exit 1
fi

ufw default deny incoming
ufw default allow outgoing
ufw allow in on tailscale0
ufw allow 41641/udp
for net in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16; do
  ufw allow proto udp from "$net" port 3702
  ufw allow proto udp from "$net" to any port 6666:6667
done
ufw --force enable

dropin="$SSHD_DROPIN_DIR/10-key-only.conf"
printf 'PasswordAuthentication no\n' >"$dropin"
chmod 0644 "$dropin"
if ! sshd -t; then
  rm -f "$dropin"
  echo "sshd -t rejected the config; drop-in removed, SSH left as it was" >&2
  exit 1
fi
# Ubuntu 24.04 can run ssh socket-activated with ssh.service inactive; then
# there is nothing to reload and the next connection reads the new config.
systemctl try-reload-or-restart ssh

loginctl enable-linger "$user"

ufw status verbose
sshd -T | grep -i '^passwordauthentication'
