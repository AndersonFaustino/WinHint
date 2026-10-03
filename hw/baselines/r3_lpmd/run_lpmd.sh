#!/usr/bin/env bash
# R3: run intel-lpmd for a run group.
#   run_lpmd.sh start [AUTO|ON]   (default AUTO: utilization-driven low-power mode)
#   run_lpmd.sh stop
#   run_lpmd.sh status
# Requires root (writable cgroup v2 cpusets) and the system D-Bus:
#   sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r3_lpmd/run_lpmd.sh start AUTO
# WINHINT_ALLOW_SYSTEM_CHANGES=1 is required. The platform check must pass first.
# Config: this CPU (F6 M186) has no model-specific file, so the generic intel_lpmd_config.xml
# is copied to $STATE_DIR with <lp_mode_cpus> = the E-cores (LPMD_LP_CPUS overrides) and passed
# with -c; LPMD_CONFIG=<file> uses a config as-is. The copy is kept for the record.
. "$(dirname "$0")/../common.sh"
PREFIX="${LPMD_PREFIX:-$WH_HWB/lpmd/prefix}"
STATE_DIR="${WH_RUN_DIR:-/tmp/winhint-lpmd}"; mkdir -p "$STATE_DIR"
PIDF="$STATE_DIR/lpmd.pid"
bin=""; for b in "$PREFIX/sbin/intel_lpmd" "$PREFIX/bin/intel_lpmd"; do [ -x "$b" ] && bin="$b"; done
ctl=""; for b in "$PREFIX/bin/intel_lpmd_control" "$PREFIX/sbin/intel_lpmd_control"; do [ -x "$b" ] && ctl="$b"; done
case "${1:-}" in
  start)
    wh_require_optin; wh_require_root; wh_require_hybrid
    "$(dirname "$0")/check_platform.sh" "$PREFIX" >&2 || wh_die "platform unsupported (see reasons above)"
    [ -n "$bin" ] || wh_die "intel_lpmd not built (install_lpmd.sh)"
    mode="${2:-AUTO}"
    pol=org.freedesktop.intel_lpmd.conf
    if [ ! -e "/etc/dbus-1/system.d/$pol" ] && [ ! -e "/usr/share/dbus-1/system.d/$pol" ]; then
      wh_die "D-Bus policy for intel_lpmd not installed; one-time opt-in step (root):
    sudo install -m 644 $PREFIX/etc/dbus-1/system.d/$pol /etc/dbus-1/system.d/ && sudo systemctl reload dbus
  (undo: sudo rm /etc/dbus-1/system.d/$pol && sudo systemctl reload dbus)"
    fi
    cfg="${LPMD_CONFIG:-}"
    if [ -z "$cfg" ]; then
      lp="${LPMD_LP_CPUS:-$(wh_ecpus)}"
      src="$PREFIX/etc/intel_lpmd/intel_lpmd_config.xml"
      [ -f "$src" ] || wh_die "$src missing (install_lpmd.sh)"
      cfg="$STATE_DIR/intel_lpmd_config.xml"
      sed "s|<lp_mode_cpus>[^<]*</lp_mode_cpus>|<lp_mode_cpus>$lp</lp_mode_cpus>|" "$src" > "$cfg"
      grep -q "<lp_mode_cpus>$lp</lp_mode_cpus>" "$cfg" || wh_die "could not set lp_mode_cpus in $cfg"
    fi
    wh_log "config $cfg (lp_mode_cpus: $(sed -n 's|.*<lp_mode_cpus>\(.*\)</lp_mode_cpus>.*|\1|p' "$cfg"))"
    nohup "$bin" --no-daemon --dbus-enable --loglevel=info -c "$cfg" > "$STATE_DIR/lpmd.log" 2>&1 &
    echo $! > "$PIDF"; sleep 2
    kill -0 "$(cat "$PIDF")" 2>/dev/null || { cat "$STATE_DIR/lpmd.log" >&2; wh_die "intel_lpmd exited"; }
    [ -n "$ctl" ] && "$ctl" "$mode" >&2 || wh_log "intel_lpmd_control not found; lpmd runs in its configured default mode"
    sleep 2; wh_log "intel_lpmd running (mode $mode)" ;;
  stop)
    wh_require_root
    [ -n "$ctl" ] && "$ctl" OFF >/dev/null 2>&1 || true
    if [ -f "$PIDF" ]; then kill -TERM "$(cat "$PIDF")" 2>/dev/null || true; sleep 2; rm -f "$PIDF"; fi
    wh_log "intel_lpmd stopped (check that cpusets were restored: cat /sys/fs/cgroup/*/cpuset.cpus.effective)" ;;
  status) [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && echo running || echo stopped ;;
  *) wh_die "usage: $0 start [AUTO|ON] | stop | status" ;;
esac
