#!/usr/bin/env bash
# R1 (stock Linux scheduler): record everything that defines the "stock" setup.
#   record_system.sh [out.txt]      (read-only; safe to run anytime)
# Captures kernel version/cmdline/config, scheduler knobs (ITMT, HFI, EAS,
# sched_ext state), cpufreq (driver, governor, EPP, turbo), SMT state, the
# P/E topology and microcode. Fields that need root are marked "(denied)".
. "$(dirname "$0")/../common.sh"
set +e
out="${1:-/dev/stdout}"
rd() { if [ -r "$1" ]; then cat "$1" 2>/dev/null || echo "(denied)"; else echo "(absent/denied)"; fi; }
uniq_vals() { for f in $1; do cat "$f" 2>/dev/null; done | sort | uniq -c | tr '\n' ';'; }
{
  echo "date: $(date -Is)"
  echo "uname: $(uname -a)"
  echo "cmdline: $(rd /proc/cmdline)"
  echo "cpu_model: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  echo "cpu_family_model: $(grep -m1 '^cpu family' /proc/cpuinfo | awk '{print $NF}')/$(grep -m1 '^model[[:space:]]*:' /proc/cpuinfo | awk '{print $NF}')"
  echo "microcode: $(grep -m1 microcode /proc/cpuinfo | awk '{print $NF}')"
  echo "flags_hybrid_hfi_itd: $(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(hybrid_cpu|hfi|itd|hwp|hwp_epp)$' | tr '\n' ' ')"
  echo "pcpus: $(wh_pcpus)   ecpus: $(wh_ecpus)"
  echo "online: $(rd /sys/devices/system/cpu/online)"
  echo "smt_control: $(rd /sys/devices/system/cpu/smt/control)  smt_active: $(rd /sys/devices/system/cpu/smt/active)"
  echo "sched_itmt_enabled: $(rd /proc/sys/kernel/sched_itmt_enabled)"
  echo "sched_energy_aware: $(rd /proc/sys/kernel/sched_energy_aware)"
  echo "sched_ext_state: $(rd /sys/kernel/sched_ext/state)  ops: $(rd /sys/kernel/sched_ext/root/ops)"
  echo "perf_event_paranoid: $(rd /proc/sys/kernel/perf_event_paranoid)"
  echo "scaling_driver: $(rd /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver)"
  echo "intel_pstate_status: $(rd /sys/devices/system/cpu/intel_pstate/status)  no_turbo: $(rd /sys/devices/system/cpu/intel_pstate/no_turbo)  hwp_dynamic_boost: $(rd /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost)"
  echo "governors: $(uniq_vals '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')"
  echo "epp: $(uniq_vals '/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference')"
  echo "max_freq_khz: $(uniq_vals '/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq')"
  echo "min_freq_khz: $(uniq_vals '/sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq')"
  echo "cpuidle_driver: $(rd /sys/devices/system/cpu/cpuidle/current_driver)  governor: $(rd /sys/devices/system/cpu/cpuidle/current_governor_ro)"
  echo "rapl_domains: $(cat /sys/class/powercap/intel-rapl*/name 2>/dev/null | tr '\n' ' ')"
  echo "rapl_energy_readable: $( [ -r /sys/class/powercap/intel-rapl:0/energy_uj ] && cat /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1 && echo yes || echo no)"
  echo "power_supply_online: $(cat /sys/class/power_supply/A*/online 2>/dev/null | tr '\n' ' ')"
  echo "loadavg: $(rd /proc/loadavg)"
  echo "--- kernel config (scheduler/power subset) ---"
  cfg=""
  for c in /proc/config.gz "/boot/config-$(uname -r)"; do [ -r "$c" ] && cfg="$c" && break; done
  if [ -z "$cfg" ]; then echo "(kernel config not readable: /proc/config.gz and /boot/config-$(uname -r) missing)"
  else { if [[ "$cfg" == *.gz ]]; then zcat "$cfg"; else cat "$cfg"; fi; } \
    | grep -E '^(CONFIG_SCHED_|CONFIG_SCHED_CLASS_EXT|CONFIG_ENERGY_MODEL|CONFIG_INTEL_HFI|CONFIG_X86_INTEL_PSTATE|CONFIG_CPU_FREQ_DEFAULT|CONFIG_HZ=|CONFIG_PREEMPT|CONFIG_NO_HZ|CONFIG_BPF_SYSCALL|CONFIG_DEBUG_INFO_BTF=|CONFIG_PERF_EVENTS_INTEL)' 2>/dev/null || true
  fi
} > "$out"
