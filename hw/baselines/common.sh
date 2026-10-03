# hw/baselines/common.sh -- shared helpers for the real-hardware baseline scripts (host).
# Source it:  . "$(dirname "$0")/../common.sh"
set -euo pipefail

wh_log()  { printf '[%s] %s\n' "$(basename "$0")" "$*" >&2; }
wh_die()  { wh_log "ERROR: $*"; exit 1; }

WH_HW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WINHINT_ROOT="${WINHINT_ROOT:-$(cd "$WH_HW_DIR/.." && pwd)}"
WINHINT_BUILD="${WINHINT_BUILD:-$WINHINT_ROOT/build}"
WH_BUILD="${WH_BUILD:-$WINHINT_BUILD}"
# Out-of-tree baseline builds (R2-R4): docs/interfaces.md §1.
WH_HWB="${WH_HWB:-$WH_BUILD/hw-baselines}"

# P/E core lists (env override like libwinhint), restricted to online CPUs
# (with SMT off the P-core siblings are offline).
wh_online_filter() {   # cpulist -> cpulist ∩ /sys/devices/system/cpu/online
  local want="$1" on c out=() a b
  on="$(cat /sys/devices/system/cpu/online 2>/dev/null || true)"
  [ -n "$want" ] || return 0
  [ -n "$on" ] || { echo "$want"; return 0; }
  declare -A ok=()
  for r in ${on//,/ }; do a=${r%-*}; b=${r#*-}; for ((c=a; c<=b; c++)); do ok[$c]=1; done; done
  for r in ${want//,/ }; do a=${r%-*}; b=${r#*-}; for ((c=a; c<=b; c++)); do [ -n "${ok[$c]:-}" ] && out+=("$c"); done; done
  (IFS=,; echo "${out[*]}")
}
wh_pcpus() { wh_online_filter "${WINHINT_PCPUS:-$(cat /sys/devices/cpu_core/cpus 2>/dev/null || true)}"; }
wh_ecpus() { wh_online_filter "${WINHINT_ECPUS:-$(cat /sys/devices/cpu_atom/cpus 2>/dev/null || true)}"; }

# Every measurement script needs an Intel hybrid CPU (P-cores = cpu_core PMU,
# E-cores = cpu_atom PMU in sysfs). Fail clearly otherwise.
wh_require_hybrid() {
  local p e
  p="$(wh_pcpus)"; e="$(wh_ecpus)"
  if [ -z "$p" ] || [ -z "$e" ]; then
    wh_die "not a hybrid P/E CPU: P-cores='${p}' E-cores='${e}' ($(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')).
  WinHint's real-hardware evaluation needs an Intel hybrid CPU exposing /sys/devices/cpu_core/cpus and
  /sys/devices/cpu_atom/cpus (e.g. Core 5 120U); override with WINHINT_PCPUS/WINHINT_ECPUS only for testing."
  fi
}

wh_require_root() {
  [ "$(id -u)" -eq 0 ] || wh_die "must run as root: re-run the exact command with sudo (see docs/guide/hardware/index.md §4)"
}

wh_require_writable_sys() {
  local f="$1"
  [ -w "$f" ] || wh_die "$f is not writable: needs root (sudo), see docs/guide/hardware/index.md §4"
}

# Refuse to do anything system-changing unless the caller explicitly opted in.
wh_require_optin() {
  [ "${WINHINT_ALLOW_SYSTEM_CHANGES:-0}" = 1 ] || wh_die \
    "this action changes system-wide state; re-run with WINHINT_ALLOW_SYSTEM_CHANGES=1 (campaign time only)"
}

# Refuse to build as root (build trees must stay owned by the user).
wh_require_user() {
  [ "$(id -u)" -ne 0 ] || wh_die "build as your normal user, not root"
}

# The conda env `winhint` must be active (tools come only from it).
wh_require_env() {
  [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = winhint ] || wh_die \
    "activate the conda env first: eval \"\$(~/.local/bin/micromamba shell hook -s bash)\" && micromamba activate winhint (or prefix with ~/.local/bin/micromamba run -n winhint)"
}

# Serialize heavy builds project-wide (docs/interfaces.md §1).
wh_heavy() { mkdir -p "$WH_BUILD"; flock "$WH_BUILD/.heavy.lock" "$@"; }
