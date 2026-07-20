#!/usr/bin/env bash
# audit-appliance.sh - read-only health audit for cctv-client appliance boxes.
#
# Detects failure modes that are invisible to `systemctl is-active` and to a
# casual `git rev-parse HEAD` — the combination that went unnoticed on
# cameraboy for three days (2026-07-17 → 20):
#
#   check 1  more than one client_agent process
#   check 2  the unit stuck in a restart loop
#   check 3  the unit will not survive a reboot (enabled + linger)
#   check 4  deployed code does not match the checkout (or was hand-patched)
#
# All are silent by construction. A unit crash-looping under
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

# Both layouts: user-mode ($HOME) and root (/opt). First match wins; a box
# only ever has one.
venv=""
for candidate in "$HOME/.local/share/cctv-client" /opt/cctv-client; do
    [ -x "$candidate/bin/python" ] && venv="$candidate" && break
done
checkout=""
for candidate in "$HOME/cctv-gpu-engine" /opt/src/cctv-gpu-engine; do
    [ -d "$candidate/.git" ] && checkout="$candidate" && break
done

install_commit=""
build_modified=""
if [ -n "$venv" ]; then
    # Ask the installed package what it is. Printing two lines rather than
    # parsing _build_info.py here keeps one implementation of the hash
    # comparison — the box's own.
    read -r install_commit build_modified <<EOF2
$("$venv/bin/python" -c 'from client_agent.build_info import resolve_build_state as r
s = r()
print(f"{s.commit or str()} {s.modified}")' 2>/dev/null)
EOF2
fi
printf 'INSTALL_COMMIT=%s\n' "$install_commit"
printf 'BUILD_MODIFIED=%s\n' "$build_modified"

checkout_commit=""
if [ -n "$checkout" ]; then
    # safe.directory: the auditor may not own the checkout (root layout, or a
    # shared box). Without it git refuses and every host reports an empty SHA,
    # which would read as "no checkout" instead of "could not read it".
    checkout_commit="$(git -c "safe.directory=$checkout" -C "$checkout" rev-parse HEAD 2>/dev/null)"
fi
printf 'CHECKOUT_COMMIT=%s\n' "$checkout_commit"
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

# check 4 - the deployed code is the code the checkout claims.
#
# `git rev-parse HEAD` in the box's checkout is how everyone asks "what is
# deployed here". That answer is only true if the installer was the last thing
# to touch site-packages. Two ways it goes wrong, both silent:
#
#   commit mismatch  — someone ran `git pull` and skipped the installer, so
#                      the checkout advertises code the box is not running
#   modified build   — someone tar/scp'd into site-packages, so BOTH SHAs
#                      agree and are both wrong; only the content hash sees it
#
# Both happened on cameraboy on 2026-07-20, the second one undetected for
# three days.
check_4() {
  local host="$1" install_commit="$2" checkout_commit="$3" modified="$4"

  if [[ -z "$install_commit" ]]; then
    warn "$host" 4 "no build record — box predates build reporting, or was not installed by install.sh/install-user.sh. Re-install to make its version knowable"
    return
  fi
  if [[ "$modified" == "True" ]]; then
    fail "$host" 4 "build zmodyfikowany po instalacji: site-packages differs from what was installed at ${install_commit:0:7}. Both SHAs still look correct — only the content hash catches this. Someone bypassed the installer; re-deploy with install-user.sh"
    return
  fi
  if [[ -z "$checkout_commit" ]]; then
    warn "$host" 4 "installed ${install_commit:0:7}, but no git checkout on the box to compare against (offline tarball install?)"
    return
  fi
  if [[ "$install_commit" != "$checkout_commit" ]]; then
    fail "$host" 4 "checkout says ${checkout_commit:0:7} but ${install_commit:0:7} is installed — someone pulled without re-running the installer, so 'git rev-parse HEAD' on this box does NOT describe what runs. Fix: ./client-appliance/install-user.sh (or install.sh for root layout)"
    return
  fi
  if [[ "$modified" != "False" ]]; then
    warn "$host" 4 "installed ${install_commit:0:7}, matches checkout, but integrity could not be verified (no recorded hash)"
    return
  fi
  pass "$host" 4 "installed ${install_commit:0:7}, matches checkout, contents unmodified"
}

for host in "${HOSTS_ARG[@]}"; do
  PROC_COUNT=""; ACTIVE_STATE=""; SUB_STATE=""; N_RESTARTS=""; UNIT_ENABLED=""; LINGER=""
  INSTALL_COMMIT=""; CHECKOUT_COMMIT=""; BUILD_MODIFIED=""
  while IFS='=' read -r key value; do
    case "$key" in
      PROC_COUNT) PROC_COUNT="$value" ;;
      ACTIVE_STATE) ACTIVE_STATE="$value" ;;
      SUB_STATE) SUB_STATE="$value" ;;
      N_RESTARTS) N_RESTARTS="$value" ;;
      UNIT_ENABLED) UNIT_ENABLED="$value" ;;
      LINGER) LINGER="$value" ;;
      INSTALL_COMMIT) INSTALL_COMMIT="$value" ;;
      CHECKOUT_COMMIT) CHECKOUT_COMMIT="$value" ;;
      BUILD_MODIFIED) BUILD_MODIFIED="$value" ;;
    esac
  done < <(probe "$host")

  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "1" ]] && check_1 "$host" "$PROC_COUNT"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "2" ]] && check_2 "$host" "$ACTIVE_STATE" "$SUB_STATE" "$N_RESTARTS"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "3" ]] && check_3 "$host" "$UNIT_ENABLED" "$LINGER"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "4" ]] &&
    check_4 "$host" "$INSTALL_COMMIT" "$CHECKOUT_COMMIT" "$BUILD_MODIFIED"
done

if [[ "$EXIT_FAIL" -eq 1 ]]; then exit 1; fi
if [[ "$EXIT_WARN" -eq 1 ]]; then exit 2; fi
exit 0
