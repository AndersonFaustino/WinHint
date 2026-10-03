#!/usr/bin/env bash
# Opt-in, temporary read access to the RAPL energy counters for one user group,
# so the campaign can measure energy WITHOUT running as root.
#
# Since CVE-2020-8694 (PLATYPUS) the kernel makes every powercap energy_uj file
# mode 0400 root. This script (run by the user, with sudo; never by the WinHint
# scripts) gives *read* access to the group of the invoking user only:
#
#   sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/rapl_access.sh grant
#   sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/rapl_access.sh revoke
#   hw/baselines/r1_stock/rapl_access.sh status                     (no root needed)
#
# grant = for every intel-rapl energy_uj:  chgrp <group of $SUDO_USER> ; chmod 0440
# revoke = chgrp root ; chmod 0400 (the kernel default). sysfs modes are not
# persistent: a reboot (or driver reload) also revokes. Exposing fine-grained
# energy to unprivileged code is a known side channel; revoke after the campaign.
# Equivalent manual commands (what `grant` does), e.g.:
#   sudo chgrp "$(id -gn)" /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj
#   sudo chmod 0440        /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj
. "$(dirname "$0")/../common.sh"
files() { ls /sys/class/powercap/intel-rapl:*/energy_uj 2>/dev/null; }
case "${1:-}" in
  status)
    for f in $(files); do
      printf '%-55s %s %s readable_by_me=%s\n' "$f" "$(stat -L -c '%A %U:%G' "$f")" \
        "$(cat "$(dirname "$f")/name" 2>/dev/null)" "$( [ -r "$f" ] && cat "$f" >/dev/null 2>&1 && echo yes || echo no)"
    done ;;
  grant)
    wh_require_optin; wh_require_root
    grp="${WINHINT_RAPL_GROUP:-$(id -gn "${SUDO_USER:-root}")}"
    for f in $(files); do chgrp "$grp" "$f"; chmod 0440 "$f"; done
    wh_log "RAPL energy_uj readable by group '$grp' until reboot (revoke: $0 revoke)" ;;
  revoke)
    wh_require_optin; wh_require_root
    for f in $(files); do chgrp root "$f"; chmod 0400 "$f"; done
    wh_log "RAPL energy_uj back to 0400 root" ;;
  *) wh_die "usage: $0 status | grant | revoke   (grant/revoke: sudo WINHINT_ALLOW_SYSTEM_CHANGES=1)" ;;
esac
