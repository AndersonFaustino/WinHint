#!/usr/bin/env bash
# R2: start/stop a sched_ext scheduler for the duration of a run group.
#
#   run_scx.sh start PRESET     (PRESET below, or "custom:<binary> <args>")
#   run_scx.sh stop
#   run_scx.sh status
#   run_scx.sh presets
#
# Requires: root, CONFIG_SCHED_CLASS_EXT kernel (7.0: yes), BPF privileges.
# i.e. sudo:  sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r2_sched_ext/run_scx.sh start bpfland_powersave
# Only one
# sched_ext scheduler can be attached system-wide; the stock scheduler returns
# when it exits. WINHINT_ALLOW_SYSTEM_CHANGES=1 is required.
. "$(dirname "$0")/../common.sh"
SCX_BIN="${SCX_BIN:-$WH_HWB/scx/target/release}"
STATE_DIR="${WH_RUN_DIR:-/tmp/winhint-scx}"
mkdir -p "$STATE_DIR"
PIDF="$STATE_DIR/scx.pid"

# Presets -- energy-oriented and performance-oriented modes of each scheduler.
# Verify flag spelling against the pinned version (install_scx.sh saves --help).
preset() {
  case "$1" in
    bpfland_powersave)  echo "scx_bpfland --primary-domain powersave" ;;
    bpfland_perf)       echo "scx_bpfland --primary-domain performance" ;;
    cosmos_powersave)   echo "scx_cosmos --primary-domain powersave" ;;
    cosmos)             echo "scx_cosmos" ;;
    lavd_powersave)     echo "scx_lavd --powersave" ;;
    lavd_autopower)     echo "scx_lavd --autopower" ;;
    lavd)               echo "scx_lavd" ;;
    custom:*)           echo "${1#custom:}" ;;
    *) return 1 ;;
  esac
}

state() { cat /sys/kernel/sched_ext/state 2>/dev/null || echo unavailable; }

case "${1:-}" in
  presets) for p in bpfland_powersave bpfland_perf cosmos_powersave cosmos lavd_powersave lavd_autopower lavd; do
             printf '%-20s %s\n' "$p" "$(preset $p)"; done ;;
  status) echo "state=$(state) ops=$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null || echo -) pid=$(cat "$PIDF" 2>/dev/null || echo -)" ;;
  start)
    wh_require_optin; wh_require_root; wh_require_hybrid
    [ -d /sys/kernel/sched_ext ] || wh_die "kernel has no sched_ext (/sys/kernel/sched_ext missing)"
    [ "$(state)" = disabled ] || wh_die "a sched_ext scheduler is already active ($(state)); stop it first"
    cmdline="$(preset "${2:-}")" || wh_die "unknown preset '${2:-}' (see: $0 presets)"
    bin="${cmdline%% *}"; args="${cmdline#"$bin"}"
    [ -x "$SCX_BIN/$bin" ] || wh_die "$SCX_BIN/$bin not built (run install_scx.sh)"
    # shellcheck disable=SC2086
    nohup "$SCX_BIN/$bin" $args > "$STATE_DIR/scx.log" 2>&1 &
    echo $! > "$PIDF"
    for _ in $(seq 1 100); do [ "$(state)" = enabled ] && break; sleep 0.1; done
    [ "$(state)" = enabled ] || { cat "$STATE_DIR/scx.log" >&2; kill "$(cat "$PIDF")" 2>/dev/null; wh_die "scheduler did not attach"; }
    sleep 2   # let it settle
    wh_log "attached: $(cat /sys/kernel/sched_ext/root/ops 2>/dev/null) ($cmdline)" ;;
  stop)
    wh_require_root
    if [ -f "$PIDF" ]; then
      kill -INT "$(cat "$PIDF")" 2>/dev/null || true
      for _ in $(seq 1 100); do [ "$(state)" = disabled ] && break; sleep 0.1; done
      rm -f "$PIDF"
    fi
    [ "$(state)" = disabled ] || [ "$(state)" = unavailable ] || wh_die "sched_ext still $(state)"
    wh_log "stock scheduler restored" ;;
  *) wh_die "usage: $0 start PRESET | stop | status | presets" ;;
esac
