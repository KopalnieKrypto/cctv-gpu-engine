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
#   check 5  ufw is off, so the client LAN reaches the panel (:8080, no login)
#   check 6  sshd accepts passwords
#   check 7  zone-annotator is not the commit, or its files not the tree, in zone-annotator.pin
#   check 8  zone-annotator is down, or listens beyond 127.0.0.1:8713
#   check 9  its unit lost a network or path restriction
#   check 10 the box's tunnel is down, or its token file is not root's and 0600
#   check 11 no outbound TCP to Cloudflare on 7844
#   check 12 the key packages ship with is missing, or not 0600 (gpu-exchange#300)
#   check 13 the host key that shipping trusts is not cctv-vps's key pinned in the repo
#
# Checks 7-13 print one SKIP line on a box without zone-annotator. Their repair is a re-run of
# install-annotator.sh, which is idempotent; this audit stays read-only for them too.
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
# Paths on the box; overridable so tests can point the real probe at a fake /etc.
UFW_CONF="${UFW_CONF:-/etc/ufw/ufw.conf}"
SSHD_CONFIG="${SSHD_CONFIG:-/etc/ssh/sshd_config}"
PIN_FILE="${PIN_FILE:-$(dirname "${BASH_SOURCE[0]}")/zone-annotator.pin}"
# Who must own the tunnel token on the box; the tests own their stub token themselves.
TOKEN_OWNER="${TOKEN_OWNER:-root}"
KNOWN_HOSTS_FILE="${KNOWN_HOSTS_FILE:-$(dirname "${BASH_SOURCE[0]}")/annotation-drop.known_hosts}"
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
skip() { printf 'SKIP  %s  checks %s: %s\n' "$1" "$2" "$3"; }

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
  "$SSH" "${SSH_FLAGS[@]}" "$host" "bash -s -- '$UNIT' '$UFW_CONF' '$SSHD_CONFIG'" 2>/dev/null <<'REMOTE'
unit="$1"
ufw_conf="$2"
sshd_main="$3"
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

# ufw.conf is world-readable, unlike the rules themselves (user.rules is 0640).
if [ ! -e "$ufw_conf" ]; then ufw_state=missing
elif [ ! -r "$ufw_conf" ]; then ufw_state=unreadable
else
    ufw_state="$(grep -m1 '^ENABLED=' "$ufw_conf" | cut -d= -f2 | tr -d "\"'")"
    ufw_state="${ufw_state:-no}"
fi
printf 'UFW_ENABLED=%s\n' "$ufw_state"

# Effective global PasswordAuthentication without root (`sshd -T` needs the
# host keys). sshd keeps the FIRST value it reads, walking Include in place,
# and everything after the first Match is conditional, so stop there.
ssh_pa=""; ssh_unreadable=""; ssh_done=""
scan_sshd() {
    local file="$1" key rest pat inc pats
    if [ ! -r "$file" ]; then ssh_unreadable="$file"; ssh_done=1; return; fi
    while IFS=$' \t=' read -r key rest || [ -n "$key" ]; do
        case "${key,,}" in
            include)
                read -ra pats <<<"$rest"
                for pat in "${pats[@]}"; do
                    [[ "$pat" == /* ]] || pat="$(dirname "$sshd_main")/$pat"
                    for inc in $pat; do
                        [ -e "$inc" ] || continue
                        scan_sshd "$inc"
                        [ -n "$ssh_done" ] && return
                    done
                done ;;
            passwordauthentication) ssh_pa="${rest%%[[:space:]]*}"; ssh_pa="${ssh_pa,,}"; ssh_done=1; return ;;
            match) ssh_done=1; return ;;
        esac
    done < "$file"
}
if [ -e "$sshd_main" ]; then
    printf 'SSHD_STATE=present\n'
    scan_sshd "$sshd_main"
else
    printf 'SSHD_STATE=missing\n'
fi
printf 'SSH_PASSWORD_AUTH=%s\n' "$ssh_pa"
printf 'SSH_UNREADABLE=%s\n' "$ssh_unreadable"

# zone-annotator and its tunnel (checks 7-11). The overrides exist for the tests only.
za_release="$(readlink "${ZA_PREFIX:-/opt/zone-annotator}/current" 2>/dev/null)"
printf 'ZA_COMMIT=%s\n' "${za_release##*/}"
# The release's files hashed as git hashes a tree, its venv left out: what the installer compared.
# In Python, because git would have to write objects somewhere and this audit writes nothing.
za_dir="$(readlink -f "${ZA_PREFIX:-/opt/zone-annotator}/current" 2>/dev/null)"
if [ -n "$za_release" ] && [ -d "$za_dir" ]; then
    printf 'ZA_TREE=%s\n' "$(python3 - "$za_dir" <<'PY'
import hashlib, os, stat, sys


def obj(kind, data):
    return hashlib.sha1(b"%s %d\0" % (kind, len(data)) + data).digest()


def tree(path, top):
    entries = []
    for name in os.listdir(path):
        if top and name == ".venv":
            continue
        full, raw = os.path.join(path, name), os.fsencode(name)
        st = os.lstat(full)
        if stat.S_ISDIR(st.st_mode):
            sha, mode = tree(full, False), b"40000"
            if sha is None:
                continue  # git keeps no empty directory
        elif stat.S_ISLNK(st.st_mode):
            sha, mode = obj(b"blob", os.fsencode(os.readlink(full))), b"120000"
        else:
            with open(full, "rb") as f:
                sha = obj(b"blob", f.read())
            mode = b"100755" if st.st_mode & stat.S_IXUSR else b"100644"
        # git orders a directory as if its name ended in a slash
        key = raw + b"/" if mode == b"40000" else raw
        entries.append((key, mode + b" " + raw + b"\0" + sha))
    if not entries and not top:
        return None
    return obj(b"tree", b"".join(entry for _, entry in sorted(entries)))


print(tree(sys.argv[1], True).hex())
PY
)"
fi
printf 'ZA_LOAD=%s\n' "$(systemctl show zone-annotator.service -p LoadState --value 2>/dev/null)"
printf 'ZA_ACTIVE=%s\n' "$(systemctl show zone-annotator.service -p ActiveState --value 2>/dev/null)"
printf 'ZA_LISTEN=%s\n' "$(ss -ltnH 'sport = :8713' 2>/dev/null | awk '{print $4}' | sort -u | tr '\n' ' ')"
systemctl show zone-annotator.service -p IPAddressDeny -p IPAddressAllow -p NoNewPrivileges \
    -p ProtectSystem -p InaccessiblePaths 2>/dev/null | sed 's/^/ZA_/'
printf 'ZT_ACTIVE=%s\n' "$(systemctl show zone-annotator-tunnel.service -p ActiveState --value 2>/dev/null)"
printf 'ZT_TOKEN_MODE=%s\n' "$(stat -c %a "${ZA_TOKEN:-/etc/zone-annotator/tunnel.token}" 2>/dev/null)"
printf 'ZT_TOKEN_OWNER=%s\n' "$(stat -c %U "${ZA_TOKEN:-/etc/zone-annotator/tunnel.token}" 2>/dev/null)"
if timeout 5 bash -c "exec 3<>/dev/tcp/${EGRESS_HOST:-region1.v2.argotunnel.com}/${EGRESS_PORT:-7844}" 2>/dev/null; then
    echo EGRESS_7844=ok
else
    echo EGRESS_7844=fail
fi
printf 'ZA_DROP_KEY_MODE=%s\n' "$(stat -c %a "$HOME/.ssh/annotation-drop" 2>/dev/null)"
# Hashed without carriage returns, as the repo copy is: a Windows checkout has them.
za_hosts="$HOME/.ssh/annotation-drop.known_hosts"
if [ -f "$za_hosts" ]; then
    printf 'ZA_DROP_HOSTS=%s\n' "$(tr -d '\r' <"$za_hosts" | sha256sum | cut -d' ' -f1)"
fi
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

# check 5 - the firewall is on.
#
# The panel binds 0.0.0.0:8080 with no login (live camera snapshots, discovery,
# RTSP test). Only ufw keeps the client's LAN off it. On 2026-11-09 cameraboy
# had ufw inactive while checks 1-4 were green.
check_5() {
  local host="$1" ufw="$2"
  case "$ufw" in
    yes) pass "$host" 5 "ufw enabled (rules not read: user.rules needs root)" ;;
    no) fail "$host" 5 "ufw disabled - the panel on :8080 (no login, live snapshots) and SSH are open to the whole client LAN. Fix: sudo client-appliance/harden-appliance.sh \$(id -un)" ;;
    missing) fail "$host" 5 "ufw not installed - nothing filters the client LAN. Fix: sudo apt-get install ufw && sudo client-appliance/harden-appliance.sh \$(id -un)" ;;
    unreadable) warn "$host" 5 "cannot read ${UFW_CONF} without sudo. Check: sudo ufw status" ;;
    "") warn "$host" 5 "could not read ufw state (host unreachable?)" ;;
    *) warn "$host" 5 "unexpected ENABLED=${ufw} in ${UFW_CONF}" ;;
  esac
}

# check 6 - SSH takes keys only.
#
# sshd defaults PasswordAuthentication to yes, so a box with no setting at all
# (cameraboy until 2026-11-09) lets anyone on the LAN guess the account
# password - and that account has sudo.
check_6() {
  local host="$1" state="$2" pa="$3" unreadable="$4"
  if [[ -z "$state" ]]; then
    warn "$host" 6 "could not read sshd config (host unreachable?)"
    return
  fi
  if [[ "$state" == "missing" ]]; then
    pass "$host" 6 "no ${SSHD_CONFIG} - SSH server not installed"
    return
  fi
  if [[ -n "$unreadable" ]]; then
    warn "$host" 6 "cannot read ${unreadable} without sudo, so the effective PasswordAuthentication is unknown. Check: sudo sshd -T | grep -i passwordauthentication"
    return
  fi
  if [[ "$pa" == "no" ]]; then
    pass "$host" 6 "SSH key-only (PasswordAuthentication no)"
    return
  fi
  fail "$host" 6 "SSH accepts passwords (PasswordAuthentication ${pa:-not set, sshd default yes}). Fix: sudo client-appliance/harden-appliance.sh \$(id -un)"
}

FIX_ANNOTATOR="sudo client-appliance/install-annotator.sh \$(id -un) zone-annotator.tar <appliance uuid>"

# check 7 - the installed zone-annotator is the pinned commit.
check_7() {
  local host="$1" commit="$2" tree="$3" pin="$4" pin_tree="$5"
  if [[ -z "$commit" ]]; then
    fail "$host" 7 "zone-annotator.service exists but zone-annotator is not installed at /opt/zone-annotator/current. Fix: $FIX_ANNOTATOR"
    return
  fi
  if [[ "$commit" != "$pin" ]]; then
    fail "$host" 7 "zone-annotator ${commit:0:7} installed, zone-annotator.pin wants ${pin:0:7}. Fix: $FIX_ANNOTATOR"
    return
  fi
  if [[ "$tree" != "$pin_tree" ]]; then
    fail "$host" 7 "zone-annotator ${commit:0:7} installed, but its files are tree ${tree:0:8}, not the pinned ${pin_tree:0:8} (changed after the install, or no python3 on the box). Fix: $FIX_ANNOTATOR"
    return
  fi
  pass "$host" 7 "zone-annotator ${commit:0:7}, its files the pinned tree"
}

# check 8 - the service runs and only the tunnel on localhost can reach it.
#
# zone-annotator has no login of its own: anything beyond 127.0.0.1 is the client LAN
# reaching it past Access.
check_8() {
  local host="$1" active="$2" listen="$3" addr wide=()
  if [[ "$active" != "active" ]]; then
    fail "$host" 8 "zone-annotator.service is ${active:-unknown}, not active. Check: journalctl -u zone-annotator -n 30; fix: $FIX_ANNOTATOR"
    return
  fi
  if [[ -z "${listen// /}" ]]; then
    fail "$host" 8 "zone-annotator.service is active but nothing listens on 127.0.0.1:8713. Fix: $FIX_ANNOTATOR"
    return
  fi
  for addr in $listen; do
    [[ "$addr" == "127.0.0.1:8713" ]] || wide+=("$addr")
  done
  if [[ ${#wide[@]} -gt 0 ]]; then
    fail "$host" 8 "port 8713 listens beyond localhost on ${wide[*]} - the client LAN reaches zone-annotator without Access. Fix: $FIX_ANNOTATOR"
    return
  fi
  pass "$host" 8 "zone-annotator active, listening only on 127.0.0.1:8713"
}

# check 9 - the effective unit still confines the tool (a drop-in can loosen what the file says).
check_9() {
  local host="$1" deny="$2" allow="$3" nnp="$4" protect="$5" inaccessible="$6" a problems=()
  [[ " $deny " == *" 0.0.0.0/0 "* && " $deny " == *" ::/0 "* ]] ||
    problems+=("IPAddressDeny=${deny:-unset}, not any")
  for a in $allow; do
    [[ "$a" == "127.0.0.0/8" || "$a" == "::1/128" ]] || problems+=("IPAddressAllow lets in $a")
  done
  [[ "$nnp" == "yes" ]] || problems+=("NoNewPrivileges=${nnp:-unset}")
  [[ "$protect" == "strict" ]] || problems+=("ProtectSystem=${protect:-unset}, not strict")
  [[ "$inaccessible" == *"/.config/cctv-client"* ]] ||
    problems+=("the agent's ~/.config/cctv-client is not in InaccessiblePaths")
  [[ "$inaccessible" == *"/.ssh"* ]] || problems+=("the agent's .ssh is not in InaccessiblePaths")
  if [[ ${#problems[@]} -gt 0 ]]; then
    local joined
    joined=$(IFS='; '; echo "${problems[*]}")
    fail "$host" 9 "zone-annotator.service is not confined: ${joined}. Fix: $FIX_ANNOTATOR (and remove any drop-in)"
    return
  fi
  pass "$host" 9 "zone-annotator confined: localhost only, no new privileges, agent config and ~/.ssh out of reach"
}

# check 10 - the tunnel is up and its token readable by root alone.
check_10() {
  local host="$1" active="$2" mode="$3" owner="$4" problems=()
  [[ "$active" == "active" ]] || problems+=("zone-annotator-tunnel.service is ${active:-unknown}")
  [[ "$mode" == "600" ]] || problems+=("token file mode ${mode:-missing}, not 600")
  [[ -z "$mode" || "$owner" == "$TOKEN_OWNER" ]] ||
    problems+=("token file owned by ${owner:-unknown}, not $TOKEN_OWNER")
  if [[ ${#problems[@]} -gt 0 ]]; then
    local joined
    joined=$(IFS='; '; echo "${problems[*]}")
    fail "$host" 10 "tunnel: ${joined}. Fix: $FIX_ANNOTATOR (no token on the box: run gpu-exchange's appliance tunnel script first)"
    return
  fi
  pass "$host" 10 "tunnel active, token file $TOKEN_OWNER's and 0600"
}

# check 11 - cloudflared can reach the edge at all; nothing in the installer can fix this.
check_11() {
  local host="$1" egress="$2"
  if [[ "$egress" == "ok" ]]; then
    pass "$host" 11 "outbound TCP 7844 to Cloudflare open"
    return
  fi
  fail "$host" 11 "no outbound TCP to Cloudflare on 7844 - the tunnel cannot connect. Open it on the client's router or firewall"
}

# check 12 - the key annotation packages leave with, readable by the agent user alone.
check_12() {
  local host="$1" mode="$2"
  if [[ -z "$mode" ]]; then
    fail "$host" 12 "shipping key ~/.ssh/annotation-drop missing - packages cannot reach cctv-vps. Fix: $FIX_ANNOTATOR, then register the key it prints"
    return
  fi
  if [[ "$mode" != "600" ]]; then
    fail "$host" 12 "shipping key ~/.ssh/annotation-drop has mode $mode, not 600. Fix: $FIX_ANNOTATOR"
    return
  fi
  pass "$host" 12 "shipping key present, 0600"
}

# check 13 - shipping trusts only cctv-vps's host key from the repo (StrictHostKeyChecking=yes).
check_13() {
  local host="$1" got="$2" want="$3"
  if [[ -z "$got" ]]; then
    fail "$host" 13 "~/.ssh/annotation-drop.known_hosts missing - packages cannot reach cctv-vps. Fix: $FIX_ANNOTATOR"
    return
  fi
  if [[ "$got" != "$want" ]]; then
    fail "$host" 13 "~/.ssh/annotation-drop.known_hosts is not the cctv-vps key pinned in client-appliance/annotation-drop.known_hosts. Fix: $FIX_ANNOTATOR"
    return
  fi
  pass "$host" 13 "cctv-vps host key as pinned in the repo"
}

for host in "${HOSTS_ARG[@]}"; do
  PROC_COUNT=""; ACTIVE_STATE=""; SUB_STATE=""; N_RESTARTS=""; UNIT_ENABLED=""; LINGER=""
  ZA_COMMIT=""; ZA_TREE=""; ZA_LOAD=""; ZA_ACTIVE=""; ZA_LISTEN=""; ZA_IPAddressDeny=""; ZA_IPAddressAllow=""
  ZA_NoNewPrivileges=""; ZA_ProtectSystem=""; ZA_InaccessiblePaths=""
  ZT_ACTIVE=""; ZT_TOKEN_MODE=""; ZT_TOKEN_OWNER=""; EGRESS_7844=""; ZA_DROP_KEY_MODE=""; ZA_DROP_HOSTS=""
  INSTALL_COMMIT=""; CHECKOUT_COMMIT=""; BUILD_MODIFIED=""
  UFW_ENABLED=""; SSHD_STATE=""; SSH_PASSWORD_AUTH=""; SSH_UNREADABLE=""
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
      UFW_ENABLED) UFW_ENABLED="$value" ;;
      SSHD_STATE) SSHD_STATE="$value" ;;
      SSH_PASSWORD_AUTH) SSH_PASSWORD_AUTH="$value" ;;
      SSH_UNREADABLE) SSH_UNREADABLE="$value" ;;
      ZA_COMMIT) ZA_COMMIT="$value" ;;
      ZA_TREE) ZA_TREE="$value" ;;
      ZA_LOAD) ZA_LOAD="$value" ;;
      ZA_ACTIVE) ZA_ACTIVE="$value" ;;
      ZA_LISTEN) ZA_LISTEN="$value" ;;
      ZA_IPAddressDeny) ZA_IPAddressDeny="$value" ;;
      ZA_IPAddressAllow) ZA_IPAddressAllow="$value" ;;
      ZA_NoNewPrivileges) ZA_NoNewPrivileges="$value" ;;
      ZA_ProtectSystem) ZA_ProtectSystem="$value" ;;
      ZA_InaccessiblePaths) ZA_InaccessiblePaths="$value" ;;
      ZT_ACTIVE) ZT_ACTIVE="$value" ;;
      ZT_TOKEN_MODE) ZT_TOKEN_MODE="$value" ;;
      ZT_TOKEN_OWNER) ZT_TOKEN_OWNER="$value" ;;
      EGRESS_7844) EGRESS_7844="$value" ;;
      ZA_DROP_KEY_MODE) ZA_DROP_KEY_MODE="$value" ;;
      ZA_DROP_HOSTS) ZA_DROP_HOSTS="$value" ;;
    esac
  done < <(probe "$host")

  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "1" ]] && check_1 "$host" "$PROC_COUNT"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "2" ]] && check_2 "$host" "$ACTIVE_STATE" "$SUB_STATE" "$N_RESTARTS"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "3" ]] && check_3 "$host" "$UNIT_ENABLED" "$LINGER"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "4" ]] &&
    check_4 "$host" "$INSTALL_COMMIT" "$CHECKOUT_COMMIT" "$BUILD_MODIFIED"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "5" ]] && check_5 "$host" "$UFW_ENABLED"
  [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "6" ]] &&
    check_6 "$host" "$SSHD_STATE" "$SSH_PASSWORD_AUTH" "$SSH_UNREADABLE"

  if [[ -z "$ZA_COMMIT" && "$ZA_LOAD" != "loaded" ]]; then
    [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" -ge 7 ]] &&
      skip "$host" 7-13 "zone-annotator not installed (a box without annotation)"
  else
    [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "7" ]] &&
      check_7 "$host" "$ZA_COMMIT" "$ZA_TREE" "$(sed -n 1p "$PIN_FILE" | tr -d '[:space:]')" \
        "$(sed -n 2p "$PIN_FILE" | tr -d '[:space:]')"
    [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "8" ]] && check_8 "$host" "$ZA_ACTIVE" "$ZA_LISTEN"
    [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "9" ]] &&
      check_9 "$host" "$ZA_IPAddressDeny" "$ZA_IPAddressAllow" "$ZA_NoNewPrivileges" \
        "$ZA_ProtectSystem" "$ZA_InaccessiblePaths"
    [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "10" ]] && check_10 "$host" "$ZT_ACTIVE" "$ZT_TOKEN_MODE" "$ZT_TOKEN_OWNER"
    [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "11" ]] && check_11 "$host" "$EGRESS_7844"
    [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "12" ]] && check_12 "$host" "$ZA_DROP_KEY_MODE"
    [[ -z "$ONLY_CHECK" || "$ONLY_CHECK" == "13" ]] &&
      check_13 "$host" "$ZA_DROP_HOSTS" "$(tr -d '\r' <"$KNOWN_HOSTS_FILE" | sha256sum | cut -d' ' -f1)"
  fi
done

if [[ "$EXIT_FAIL" -eq 1 ]]; then exit 1; fi
if [[ "$EXIT_WARN" -eq 1 ]]; then exit 2; fi
exit 0
