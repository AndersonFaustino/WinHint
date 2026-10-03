#!/usr/bin/env bash
# R3: check whether intel-lpmd (github.com/intel/intel-lpmd) can be used here.
#   check_platform.sh [LPMD_PREFIX]      read-only; prints SUPPORTED/UNSUPPORTED + reasons
# Exit 0 = supported, 1 = unsupported (the reasons go into the paper's
# "dropped baseline" note if so).
. "$(dirname "$0")/../common.sh"
set +e
PREFIX="${1:-$WH_HWB/lpmd/prefix}"
ok=1; reasons=()
vendor=$(grep -m1 vendor_id /proc/cpuinfo | awk '{print $NF}')
fam=$(grep -m1 '^cpu family' /proc/cpuinfo | awk '{print $NF}')
model=$(grep -m1 '^model[[:space:]]*:' /proc/cpuinfo | awk '{print $NF}')
echo "cpu: $vendor family $fam model $model ($(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-))"
[ "$vendor" = GenuineIntel ] || { ok=0; reasons+=("not an Intel CPU"); }
[ -d /sys/devices/cpu_core ] && [ -d /sys/devices/cpu_atom ] || { ok=0; reasons+=("not a hybrid CPU (no cpu_core/cpu_atom PMUs)"); }
# Platforms with a dedicated config in intel-lpmd (data/intel_lpmd_config_F6_M<model>*.xml);
# others use the generic config, whose low-power cpu set must be given manually.
cfg=$(ls "$PREFIX"/etc/intel_lpmd/intel_lpmd_config_F${fam}_M${model}*.xml 2>/dev/null | head -1)
src_cfg=$(ls "$WH_HWB"/intel-lpmd/data/intel_lpmd_config_F${fam}_M${model}*.xml 2>/dev/null | head -1)
if [ -n "$cfg$src_cfg" ]; then echo "model-specific config: ${cfg:-$src_cfg}"
else echo "model-specific config: none (generic intel_lpmd_config.xml; set lp_mode_cpus to the E-cores: $(wh_ecpus))"; fi
# Mode switching needs cgroup v2 cpuset (or the powerclamp idle-injection path).
if grep -q cgroup2 /proc/mounts; then
  ctl=$(cat /sys/fs/cgroup/cgroup.controllers 2>/dev/null)
  [[ " $ctl " == *" cpuset "* ]] && echo "cgroup v2 cpuset: yes" || { ok=0; reasons+=("cgroup v2 cpuset controller not available at the root"); }
  [ -w /sys/fs/cgroup/cgroup.subtree_control ] || reasons+=("(note) /sys/fs/cgroup not writable by this user: lpmd must run as root (sudo)")
else ok=0; reasons+=("no cgroup v2"); fi
[ -e /sys/devices/system/cpu/intel_pstate/status ] || reasons+=("(note) intel_pstate not present")
grep -qs intel_powerclamp /sys/class/thermal/cooling_device*/type && echo "intel_powerclamp: present" || echo "intel_powerclamp: absent (idle-injection mode unavailable; cgroup mode only)"
# HFI notifications (optional input of lpmd)
grep -qw hfi /proc/cpuinfo && echo "HFI: yes" || echo "HFI: no"
# D-Bus is needed by intel_lpmd_control
[ -S /run/dbus/system_bus_socket ] && echo "system D-Bus: yes" || reasons+=("(note) no system D-Bus socket here: lpmd needs the system bus for intel_lpmd_control")
[ -x "$PREFIX/sbin/intel_lpmd" ] || [ -x "$PREFIX/bin/intel_lpmd" ] && echo "intel_lpmd: built in $PREFIX" || reasons+=("(note) intel_lpmd not built yet (install_lpmd.sh)")
for r in "${reasons[@]}"; do echo "reason: $r"; done
if [ $ok = 1 ]; then echo SUPPORTED; exit 0; else echo UNSUPPORTED; exit 1; fi
