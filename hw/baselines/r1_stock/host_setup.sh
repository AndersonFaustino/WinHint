#!/usr/bin/env bash
# Campaign-time system configuration (governor/EPP/turbo/SMT) with save+restore.
# NEVER run implicitly: requires WINHINT_ALLOW_SYSTEM_CHANGES=1 and root, e.g.
#   sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/host_setup.sh apply --smt off
#
#   host_setup.sh save    STATE_FILE
#   host_setup.sh apply   [--governor performance|powersave] [--epp VALUE]
#                         [--no-turbo 0|1] [--smt on|off]
#   host_setup.sh restore STATE_FILE
#
# Recommended fixed setup for the campaign (documented in docs/guide/hardware/index.md):
#   --governor performance --epp performance --no-turbo 1   (stable frequency)
# or, to keep turbo but fix the policy:  --governor powersave --epp balance_performance
. "$(dirname "$0")/../common.sh"
cmd="${1:-}"; shift || true
CPUFREQ=/sys/devices/system/cpu
case "$cmd" in
  save)
    st="${1:?state file}"
    {
      echo "smt=$(cat $CPUFREQ/smt/control)"
      echo "no_turbo=$(cat $CPUFREQ/intel_pstate/no_turbo 2>/dev/null || echo NA)"
      for g in $CPUFREQ/cpu[0-9]*/cpufreq; do
        c=$(basename "$(dirname "$g")")
        echo "gov_$c=$(cat "$g/scaling_governor" 2>/dev/null || echo NA)"
        echo "epp_$c=$(cat "$g/energy_performance_preference" 2>/dev/null || echo NA)"
      done
    } > "$st"
    wh_log "saved system state to $st" ;;
  apply)
    wh_require_optin; wh_require_root
    gov=""; epp=""; nt=""; smt=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --governor) gov="$2"; shift 2 ;;
        --epp) epp="$2"; shift 2 ;;
        --no-turbo) nt="$2"; shift 2 ;;
        --smt) smt="$2"; shift 2 ;;
        *) wh_die "unknown option $1" ;;
      esac
    done
    if [ -n "$smt" ]; then
      wh_require_writable_sys $CPUFREQ/smt/control
      echo "$smt" > $CPUFREQ/smt/control; wh_log "SMT -> $smt"; sleep 1
    fi
    if [ -n "$nt" ]; then
      wh_require_writable_sys $CPUFREQ/intel_pstate/no_turbo
      echo "$nt" > $CPUFREQ/intel_pstate/no_turbo; wh_log "no_turbo -> $nt"
    fi
    for g in $CPUFREQ/cpu[0-9]*/cpufreq; do
      [ -d "$g" ] || continue
      if [ -n "$gov" ]; then wh_require_writable_sys "$g/scaling_governor"; echo "$gov" > "$g/scaling_governor"; fi
      if [ -n "$epp" ] && [ -w "$g/energy_performance_preference" ]; then
        echo "$epp" > "$g/energy_performance_preference" 2>/dev/null || wh_log "EPP $epp rejected on $g (performance governor pins EPP)"
      fi
    done
    [ -n "$gov" ] && wh_log "governor -> $gov"; [ -n "$epp" ] && wh_log "EPP -> $epp"; true ;;
  restore)
    wh_require_optin; wh_require_root
    st="${1:?state file}"
    smt=$(grep '^smt=' "$st" | cut -d= -f2)
    case "$smt" in on|off) echo "$smt" > $CPUFREQ/smt/control; sleep 1 ;; esac
    nt=$(grep '^no_turbo=' "$st" | cut -d= -f2)
    [ "$nt" != NA ] && echo "$nt" > $CPUFREQ/intel_pstate/no_turbo
    for g in $CPUFREQ/cpu[0-9]*/cpufreq; do
      c=$(basename "$(dirname "$g")")
      v=$(grep "^gov_$c=" "$st" | cut -d= -f2); [ -n "$v" ] && [ "$v" != NA ] && echo "$v" > "$g/scaling_governor"
      v=$(grep "^epp_$c=" "$st" | cut -d= -f2); [ -n "$v" ] && [ "$v" != NA ] && echo "$v" > "$g/energy_performance_preference" 2>/dev/null
    done
    wh_log "restored system state from $st" ;;
  *) wh_die "usage: $0 save FILE | apply [opts] | restore FILE" ;;
esac
