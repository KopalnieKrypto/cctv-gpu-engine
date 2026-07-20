#!/usr/bin/env bash
# audit-appliance.sh - read-only health audit for cctv-client appliance boxes.
#
# Detects the two failure modes that are invisible to `systemctl is-active`
# and therefore went unnoticed on cameraboy for three days (2026-07-17 → 20):
#
#   check 1  more than one client_agent process
#   check 2  the unit stuck in a restart loop
#   check 3  the unit will not survive a reboot (enabled + linger)
#
# All three are silent by construction. A unit crash-looping under
# `Restart=on-failure` reports ActiveState=activating / SubState=auto-restart,
# never `failed`, so `is-active` prints "activating" and any naive
# `is-active | grep -q active` check passes. Meanwhile a stray hand-launched
# process holds :8080, the unit's copy dies on bind, and the box keeps serving
# production from whatever code the stray imported - possibly days stale.
# Nothing else guards against a double-run: the port bind is the only interlock.
#
# Usage:
#   client-appliance/audit-appliance.sh cameraboy
#   HOSTS="cameraboy other-box" client-appliance/audit-appliance.sh
#   client-appliance/audit-appliance.sh --check 1 cameraboy
#
# Exit codes: 0 = all PASS, 1 = any FAIL, 2 = WARN only.
#
# Read-only: this script never mutates a box. Remediation is deliberately
# manual - stopping the *wrong* process (the one actually serving production)
# mid-recording is worse than the drift it fixes, and picking correctly needs
# a human to read the two start times. See README "Recovering from a double-run".
#
# Mocking: `ssh` is invoked via $SSH (defaults to `ssh`). Tests override it.

# `set -e` is intentionally OFF, matching gpu-exchange's audit-gpu-server.sh:
# every check captures its own failure and continues so one unreachable box
# doesn't abort the fan-out. Keep -u and pipefail.
set -uo pipefail

SSH="${SSH:-ssh}"
SSH_FLAGS=(-o BatchMode=yes -o ConnectTimeout=10)

# NRestarts at or above this is a loop, not a blip. A one-off restart after an
# OOM or a transient platform 502 is normal operation and must not cry wolf;
# cameraboy's real incident sat at 4484.
RESTART_LOOP_THRESHOLD="${RESTART_LOOP_THRESHOLD:-3}"

UNIT="${UNIT:-cctv-client}"
ONLY_CHECK=""
HOSTS_ARG=()

usage() {
  cat >&2 <<EOF
usage: $0 [--check N] [HOST ...]
   or: HOSTS="h1 h2" $0 [--check N]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) ONLY_CHECK="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; usage; exit 64 ;;
    *) HOSTS_ARG+=("$1"); shift ;;
  esac
done

if [[ ${#HOSTS_ARG[@]} -eq 0 ]]; then
  read -r -a HOSTS_ARG <<<"${HOSTS:-}"
fi
if [[ ${#HOSTS_ARG[@]} -eq 0 ]]; then
  echo "no host given" >&2
  usage
  exit 64
fi

EXIT_FAIL=0
EXIT_WARN=0

pass() { printf 'PASS  %s  check %s: %s\n' "$1" "$2" "$3"; }
warn() { printf 'WARN  %s  check %s: %s\n' "$1" "$2" "$3"; EXIT_WARN=1; }
fail() { printf 'FAIL  %s  check %s: %s\n' "$1" "$2" "$3"; EXIT_FAIL=1; }

# One round trip per host: probe everything, parse locally. Keeps the remote
# surface tiny (easy to mock, easy to audit) and avoids N ssh handshakes.
#
# The bracket in 'client_agent[.]appliance' stops the pattern from matching the
# probe's own command line - the remote shell's argv contains the bracketed
# form, which does not match the literal it expands to. Without it every box
# reports one phantom process. (Same trap as `pkill -f client_agent` over ssh,
# which matches its own ssh command and kills the session.)
# The remote script is fed on stdin from a QUOTED heredoc and the unit name is
# passed as a positional arg. Nothing is interpolated locally, so nested
# command substitution (`$(id -un)` inside a quoted argument) survives intact.
#
# The earlier version inlined this as a double-quoted ssh argument and the
# escaping mangled `$(id -un)`, making the linger probe return empty on every
# box — a false FAIL on a host that was correctly configured. Keep it on stdin.
#
# Bonus: the pattern no longer appears in the remote *command line* at all
# (it arrives via stdin), so there is nothing for pgrep to self-match on.
probe() {
  local host="$1"
  "$SSH" "${SSH_FLAGS[@]}" "$host" "bash -s -- '$UNIT'" 2>/dev/null <<'REMOTE'
unit="$1"
printf 'PROC_COUNT=%s\n' "$(pgrep -fc 'client_agent[.]appliance' 2>/dev/null || echo 0)"
printf 'ACTIVE_STATE=%s\n' "$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null)"
printf 'SUB_STATE=%s\n' "$(systemctl --user show "$unit" -p SubState --value 2>/dev/null)"
printf 'N_RESTARTS=%s\n' "$(systemctl --user show "$unit" -p NRestarts --value 2>/dev/null)"
printf 'UNIT_ENABLED=%s\n' "$(systemctl --user is-enabled "$unit" 2>/dev/null)"
printf 'LINGER=%s\n' "$(loginctl show-user "$(id -un)" --property=Linger --value 2>/dev/null)"
REMOTE
}

# check 1 - exactly one client_agent process.
#
# >1 is the double-run: a hand-launched stray alongside the unit. Which one
# serves production is then decided by whichever won the port bind, and the
# loser crash-loops forever behind a green-looking `is-active`.
# 0 means nothing is running at all, which no other check here would catch.
check_1() {
  local host="$1" count="$2"
  if [[ -z "$count" || ! "$count" =~ ^[0-9]+$ ]]; then
    warn "$host" 1 "could not read process count (host unreachable?)"
    return
  fi
  case "$count" in
    1) pass "$host" 1 "exactly one client_agent process" ;;
    0) fail "$host" 1 "no client_agent process running" ;;
    *) fail "$host" 1 "$count client_agent processes - double-run; a stray may be serving production from stale code. Do NOT pkill -f: identify the systemd MainPID, then kill the OTHER by exact PID" ;;
  esac
}

# check 2 - unit is not stuck in a restart loop.
#
# SubState=auto-restart is the smoking gun `is-active` hides: the unit is
# mid-loop, dying and respawning every RestartSec. NRestarts catches the same
# condition when the probe happens to land during an up-swing.
check_2() {
  local host="$1" active="$2" sub="$3" restarts="$4"
  if [[ -z "$active" ]]; then
    warn "$host" 2 "could not read unit state (unit missing or host unreachable?)"
    return
  fi
  if [[ ! "$restarts" =~ ^[0-9]+$ ]]; then
    restarts=0
  fi
  if [[ "$sub" == "auto-restart" ]]; then
    fail "$host" 2 "unit in restart loop (ActiveState=$active SubState=auto-restart, NRestarts=$restarts) - is-active reports '$active', never 'failed', so this passes a naive health check. Check: journalctl --user -u ${UNIT} -n 30"
    return
  fi
  if [[ "$restarts" -ge "$RESTART_LOOP_THRESHOLD" ]]; then
    fail "$host" 2 "unit restarted $restarts times (threshold $RESTART_LOOP_THRESHOLD) - crash-looping. Check: journalctl --user -u ${UNIT} -n 30"
    return
  fi
  if [[ "$active" != "active" ]]; then
    warn "$host" 2 "unit not active (ActiveState=$active SubState=$sub)"
    return
  fi
  if [[ "$restarts" -gt 0 ]]; then
    warn "$host" 2 "unit active but restarted $restarts time(s) - below loop threshold, worth a glance"
    return
  fi
  pass "$host" 2 "unit active, no restarts"
}

# check 3 - the unit will actually come back after a reboot.
#
# A *user* unit needs BOTH `is-enabled` and lingering. Without linger, systemd
# tears the user manager down at logout and only rebuilds it on the next login
# - so an enabled unit on a headless box simply never starts, and the box looks
# fine right up until it reboots. Unattended-upgrades reboots these boxes on
# their own schedule, so "fine until reboot" means "fine until some night".
#
# This is not hypothetical: before the unit existed, cameraboy ran from a
# `nohup` one-liner and a reboot cost an ~18 h outage (2026-07-14 → 16), during
# which a queued task aged its footage out of the 1 h rolling buffer and could
# never be recovered. Nothing on the platform reaps a stale `queued` task, so
# it hangs invisibly rather than failing loudly.
check_3() {
  local host="$1" enabled="$2" linger="$3"
  if [[ -z "$enabled" && -z "$linger" ]]; then
    warn "$host" 3 "could not read enablement/linger (host unreachable?)"
    return
  fi
  local problems=()
  [[ "$enabled" != "enabled" ]] && problems+=("unit is '${enabled:-unknown}', not enabled")
  [[ "$linger" != "yes" ]] && problems+=("Linger=${linger:-unknown}, not yes")
  if [[ ${#problems[@]} -eq 0 ]]; then
    pass "$host" 3 "survives reboot (unit enabled, linger on)"
    return
  fi
  local joined
  joined=$(IFS='; '; echo "${problems[*]}")
  fail "$host" 3 "will NOT survive reboot: ${joined}. A user unit needs both. Fix: systemctl --user enable ${UNIT}; sudo loginctl enable-linger \$(id -un)"
}

for host in "${HOSTS_ARG[@]}"; do
  PROC_COUNT=""; ACTIVE_STATE=""; SUB_STATE=""; N_RESTARTS=""; UNIT_ENABLED=""; LINGER=""
  while IFS='=' read -r key value; do
    case "$key" in
      PROC_COUNT) PROC_COUNT="$value" ;;
      ACTIVE_STATE) ACTIVE_STATE="$value" ;;
      SUB_STATE) SUB_STATE="$value" ;;
      N_RESTARTS) N_RESTARTS="$value" ;;
      UNIT_ENABLED) UNIT_ENABLED="$value" ;;
      LINGER) LINGER="$value" ;;
    esac
  done < <(probe "$host")

  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "1" ]] && check_1 "$host" "$PROC_COUNT"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "2" ]] && check_2 "$host" "$ACTIVE_STATE" "$SUB_STATE" "$N_RESTARTS"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "3" ]] && check_3 "$host" "$UNIT_ENABLED" "$LINGER"
done

if [[ "$EXIT_FAIL" -eq 1 ]]; then exit 1; fi
if [[ "$EXIT_WARN" -eq 1 ]]; then exit 2; fi
exit 0
