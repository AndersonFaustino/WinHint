#!/usr/bin/env bash
# R3: build intel-lpmd (github.com/intel/intel-lpmd) on the host with the conda env
# `winhint` toolchain. Opt-in, never run automatically; no root, no system-wide install.
#
#   micromamba run -n winhint bash hw/baselines/r3_lpmd/install_lpmd.sh
#
# Uses upstream's Makefile.simple (no autotools needed; LPMD_BUILD=autotools for
# ./autogen.sh). Installs into $WINHINT_BUILD/hw-baselines/lpmd/prefix, including the
# D-Bus policy and systemd unit (NOT copied to /etc). Pin with LPMD_REF.
#
# Build dependencies, all from the conda env (environment.yml): glib-2.0 gio-2.0
# gio-unix-2.0 gmodule-2.0 (+ glib-compile-resources and glibconfig.h from conda-forge `glib`)
# libxml-2.0 libnl-3.0 libnl-genl-3.0 libsystemd. One gap remains:
#   * upower-glib (no conda package as of 2026-10): LPMD_UPOWER=auto (default) uses the real
#     library if pkg-config finds it, else the stub in upower_stub/ (= "upowerd unreachable,
#     on AC power"; lpmd only uses upower to detect battery mode). LPMD_UPOWER=real refuses
#     the stub. The choice is recorded in $PREFIX/WINHINT_LPMD_DEVIATIONS.
# Running lpmd needs root (cgroup cpusets, D-Bus name): see run_lpmd.sh / docs/guide/hardware/index.md.
. "$(dirname "$0")/../common.sh"
wh_require_user
wh_require_env
LPMD_REF="${LPMD_REF:-main}"
LPMD_UPOWER="${LPMD_UPOWER:-auto}"
SRC="$WH_HWB/intel-lpmd"; PREFIX="$WH_HWB/lpmd/prefix"
STUB_SRC="$(cd "$(dirname "$0")" && pwd)/upower_stub"; STUB="$WH_HWB/lpmd/upower-stub"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
# Only the env's .pc files: never pick up host (apt) libraries by accident.
export PKG_CONFIG_LIBDIR="$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/share/pkgconfig"
missing=()
for m in glib-2.0 gio-2.0 gio-unix-2.0 gmodule-2.0 libxml-2.0 libnl-3.0 libnl-genl-3.0 libsystemd; do
  pkg-config --exists "$m" || missing+=("$m")
done
[ ${#missing[@]} -eq 0 ] || wh_die "missing build deps in the conda env: ${missing[*]} -- R3 cannot be built with conda tools (report it; do not apt-install)"
mkdir -p "$WH_HWB/lpmd"
deviations=()

# --- glib developer bits: conda-forge `glib` (environment.yml) provides
# glib-compile-resources and lib/glib-2.0/include/glibconfig.h. GLIB_COMPILE_RESOURCES overrides.
GCR="${GLIB_COMPILE_RESOURCES:-$(command -v glib-compile-resources || true)}"
[ -n "$GCR" ] && [ -x "$GCR" ] && [ -f "$CONDA_PREFIX/lib/glib-2.0/include/glibconfig.h" ] || wh_die \
  "the env lacks conda-forge 'glib' (glib-compile-resources/glibconfig.h): run tooling/create_conda_env.sh --update"
export PATH="$(dirname "$GCR"):$PATH"   # Makefile.simple calls glib-compile-resources by name

# --- upower-glib: real (if a conda package ever appears) or the stub
if pkg-config --exists upower-glib && [ "$LPMD_UPOWER" != stub ]; then
  wh_log "upower-glib: real library ($(pkg-config --modversion upower-glib))"
else
  [ "$LPMD_UPOWER" != real ] || wh_die "upower-glib not in the conda env and LPMD_UPOWER=real"
  wh_log "upower-glib has no conda package: building the stub from $STUB_SRC (battery state not monitored = AC power)"
  mkdir -p "$STUB/include" "$STUB/lib/pkgconfig"
  cp "$STUB_SRC/upower.h" "$STUB/include/"
  # shellcheck disable=SC2046
  ${CC:-cc} -O2 -fPIC -c "$STUB_SRC/upower_stub.c" -I"$STUB/include" $(pkg-config --cflags gio-2.0) \
    -o "$STUB/lib/upower_stub.o"
  ar rcs "$STUB/lib/libupower-glib.a" "$STUB/lib/upower_stub.o"
  cat > "$STUB/lib/pkgconfig/upower-glib.pc" <<EOF
prefix=$STUB
Name: upower-glib
Description: WinHint stub of upower-glib (no conda package): reports AC power, no upowerd
Version: 0.0-winhint-stub
Requires: glib-2.0 gobject-2.0 gio-2.0
Cflags: -I\${prefix}/include
Libs: \${prefix}/lib/libupower-glib.a
EOF
  export PKG_CONFIG_LIBDIR="$STUB/lib/pkgconfig:$PKG_CONFIG_LIBDIR"
  deviations+=("upower-glib replaced by hw/baselines/r3_lpmd/upower_stub (no conda package): lpmd assumes AC power and does not follow battery/AC changes; run the campaign on AC power")
fi

[ -d "$SRC/.git" ] || git clone https://github.com/intel/intel-lpmd "$SRC"
git -C "$SRC" fetch --tags origin || wh_log "WARNING: git fetch failed (offline?), building the checked-out tree"
git -C "$SRC" checkout "$LPMD_REF"
cd "$SRC"
# A small C build (-j1, like every non-heavy compile in docs/interfaces.md §1).
if [ "${LPMD_BUILD:-simple}" = autotools ]; then
  for t in autoreconf automake libtoolize; do command -v "$t" >/dev/null || wh_die "$t not in the env"; done
  { ./autogen.sh --prefix="$PREFIX" --sysconfdir="$PREFIX/etc" --localstatedir="$PREFIX/var" &&
    make -j1 CFLAGS="-O2" && make install; } 2>&1 | tee "$WH_HWB/lpmd-build.log"
else
  { make -f Makefile.simple clean srctree="$SRC" >/dev/null && make -f Makefile.simple -j1 CC="${CC:-cc}" srctree="$SRC" prefix="$PREFIX" \
      LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib" &&
    make -f Makefile.simple install srctree="$SRC" prefix="$PREFIX" \
      systemd_unitdir="$PREFIX/lib/systemd/system" dbus_sysdir="$PREFIX/etc/dbus-1/system.d"; } \
    2>&1 | tee "$WH_HWB/lpmd-build.log"
fi
[ "${PIPESTATUS[0]}" -eq 0 ] && [ -x "$PREFIX/sbin/intel_lpmd" ] || wh_die "intel-lpmd build failed (see $WH_HWB/lpmd-build.log)"
git rev-parse HEAD > "$PREFIX/WINHINT_LPMD_COMMIT"
{ echo "# intel-lpmd @ $(cat "$PREFIX/WINHINT_LPMD_COMMIT"), built $(date -Is) with the conda env winhint"
  for d in "${deviations[@]}"; do echo "- $d"; done; } > "$PREFIX/WINHINT_LPMD_DEVIATIONS"
wh_log "intel-lpmd @ $(cat "$PREFIX/WINHINT_LPMD_COMMIT") installed in $PREFIX (deviations: $PREFIX/WINHINT_LPMD_DEVIATIONS); now run check_platform.sh"
