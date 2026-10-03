#!/usr/bin/env bash
# R4 backend: PMCTrack (github.com/jcsaezal/pmctrack), reused as-is.
#
#   install_pmctrack.sh build     # user-space tools (libpmctrack, pmctrack CLI)   -- no root
#   install_pmctrack.sh module    # kernel module, built against the host headers -- no root
#   install_pmctrack.sh load      # insmod (root + WINHINT_ALLOW_SYSTEM_CHANGES=1; documented, never automatic)
#   install_pmctrack.sh unload    # rmmod  (root + WINHINT_ALLOW_SYSTEM_CHANGES=1)
#   install_pmctrack.sh status
#
# Everything is built under $WINHINT_BUILD/hw-baselines/pmctrack with the conda env
# `winhint` compilers (run inside `micromamba run -n winhint`). The module is built
# against /lib/modules/$(uname -r)/build (the running kernel; upstream hardcodes it). The kernel's
# own build helpers (scripts/, objtool) come from the host headers package; the C
# compiler is the env's gcc unless KCC is given (a module must be built with a
# compiler close to the one that built the kernel: see /proc/version).
# Loading modifies the running kernel, so it is opt-in and only ever done by the
# user with sudo:
#   sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r4_pie/install_pmctrack.sh load
# If the module does not build/load on kernel 7.0, R4 uses its perf_event_open
# backend (pie_daemon -b perf), which implements the same policy.
. "$(dirname "$0")/../common.sh"
PMC_REF="${PMC_REF:-master}"
SRC="$WH_HWB/pmctrack"
KDIR="${KDIR:-/lib/modules/$(uname -r)/build}"
MODDIR="$SRC/src/modules/pmcs/intel-core"
CLI="$SRC/bin/pmctrack"
fetch() {
  mkdir -p "$WH_HWB"
  [ -d "$SRC/.git" ] || git clone https://github.com/jcsaezal/pmctrack "$SRC"
  git -C "$SRC" checkout "$PMC_REF"
  git -C "$SRC" rev-parse HEAD > "$SRC/WINHINT_PMCTRACK_COMMIT"
}
find_ko() { find "$MODDIR" -name 'mchw_intel_core.ko' 2>/dev/null | head -1; }
case "${1:-}" in
  build)
    wh_require_user; wh_require_env; fetch
    # -j1 per project rules; heavy lock serializes with other builds.
    # Upstream links the CLI with -static; the conda sysroot may lack a static libc,
    # so fall back to a dynamic link (PMCTRACK_STATIC=0 forces it).
    ldf="-L../../lib/libpmctrack -lpmctrack"; [ "${PMCTRACK_STATIC:-1}" = 1 ] && ldf="$ldf -static"
    ( cd "$SRC" && export PMCTRACK_ROOT="$SRC" && \
      wh_heavy make -C src/lib/libpmctrack -j1 CC="${CC:-gcc}" && \
      { wh_heavy make -C src/cmdtools/pmctrack -j1 CC="${CC:-gcc}" LDFLAGS="$ldf" || \
        wh_heavy make -C src/cmdtools/pmctrack -j1 CC="${CC:-gcc}" \
          LDFLAGS="-L../../lib/libpmctrack -l:libpmctrack.a"; } ) 2>&1 | tee "$SRC/build-user.log"
    [ -x "$CLI" ] || wh_die "user-space build failed (see $SRC/build-user.log)"
    wh_log "pmctrack CLI: $CLI (run with PMCTRACK_ROOT=$SRC; needs the module loaded)" ;;
  module)
    wh_require_user; wh_require_env
    [ -d "$KDIR" ] || wh_die "kernel headers for $(uname -r) missing ($KDIR)"
    fetch
    kcc="${KCC:-$(command -v x86_64-conda-linux-gnu-gcc || command -v gcc)}"
    wh_log "building mchw_intel_core.ko against $KDIR with $kcc (kernel built by: $(cut -d'(' -f3- /proc/version | cut -d')' -f1))"
    # Upstream Makefile: symlinks the sources, then make -C /lib/modules/$(uname -r)/build M=$PWD modules.
    # CC on the command line propagates to the kbuild sub-make; HOSTCC stays the kernel's tools.
    # Upstream passes its include paths/defines in EXTRA_CFLAGS, which Kbuild no longer
    # honours (removed in recent kernels; 7.0 ignores it -> "pmc/pmu_config.h: No such
    # file"). KCFLAGS carries the same flags into the module build (upstream unchanged).
    kcflags="-DCONFIG_PMC_CORE_I7 -I$MODDIR/../include -I$MODDIR/.. -DHOST$(hostname | tr -c 'A-Za-z0-9_\n' _)"
    ( cd "$MODDIR" && wh_heavy make -j1 CC="$kcc" KCFLAGS="$kcflags" ) 2>&1 | tee "$SRC/build-module.log"
    ko=$(find_ko)
    [ -n "$ko" ] || wh_die "module build failed (kernel $(uname -r) may be unsupported by PMCTrack; see $SRC/build-module.log)"
    wh_log "built $ko (NOT loaded). To load: sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 $0 load" ;;
  load)
    wh_require_optin; wh_require_root
    ko=$(find_ko); [ -n "$ko" ] || wh_die "module not built (run: $0 module)"
    insmod "$ko" && wh_log "loaded $ko; /proc/pmc should now exist. Unload with: $0 unload" ;;
  unload)
    wh_require_optin; wh_require_root
    rmmod mchw_intel_core && wh_log "unloaded mchw_intel_core" ;;
  status)
    echo "src: $SRC ($(cat "$SRC/WINHINT_PMCTRACK_COMMIT" 2>/dev/null || echo not fetched))"
    echo "cli: $( [ -x "$CLI" ] && echo "$CLI" || echo not built)"
    echo "module: $(find_ko)"
    echo "loaded: $([ -e /proc/pmc ] && echo yes || echo no)" ;;
  *) wh_die "usage: $0 build | module | load | unload | status" ;;
esac
