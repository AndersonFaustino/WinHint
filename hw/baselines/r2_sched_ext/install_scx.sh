#!/usr/bin/env bash
# R2: build the sched_ext schedulers (scx_bpfland, scx_cosmos, scx_lavd) from
# github.com/sched-ext/scx on the host with the conda env `winhint` toolchain
# (rust/cargo, clang for the BPF objects, libbpf/libelf/zlib/zstd headers).
# Opt-in, never run automatically; building needs network (git + crates.io), no root.
#
#   micromamba run -n winhint bash hw/baselines/r2_sched_ext/install_scx.sh
#
# Runs under the project heavy-build lock with cargo -j1. Pin the version with
# SCX_REF (tag/commit; the checked-out commit is saved). Outputs:
#   $WINHINT_BUILD/hw-baselines/scx/target/release/scx_{bpfland,cosmos,lavd}
#   $WINHINT_BUILD/hw-baselines/scx/{WINHINT_SCX_COMMIT,scx_*.help.txt,build.log}
# Running a scheduler needs root (BPF + sched_ext attach): see run_scx.sh / docs/guide/hardware/index.md.
. "$(dirname "$0")/../common.sh"
wh_require_user
wh_require_env
SCX_REF="${SCX_REF:-main}"        # pin a release tag for the paper (recorded in WINHINT_SCX_COMMIT)
SCX_DIR="$WH_HWB/scx"
SCHEDS="${SCX_SCHEDS:-scx_bpfland scx_cosmos scx_lavd}"
for t in cargo rustc clang git; do command -v "$t" >/dev/null || wh_die "$t not found in the env"; done
mkdir -p "$WH_HWB"
export CARGO_HOME="${CARGO_HOME:-$WH_HWB/cargo-home}"   # crates cache under build/, not ~/.cargo
export CARGO_TARGET_DIR="$SCX_DIR/target"
export CARGO_BUILD_JOBS=1
# Use the env's libraries/headers (libbpf-sys/libelf/zlib/zstd) and libclang for bindgen.
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LIBCLANG_PATH="${LIBCLANG_PATH:-$CONDA_PREFIX/lib}"
export BPF_CLANG="${BPF_CLANG:-$(command -v clang)}"
export CC="${CC:-$(command -v clang)}"
export CFLAGS="${CFLAGS:-} -I$CONDA_PREFIX/include"
export LDFLAGS="${LDFLAGS:-} -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export RUSTFLAGS="${RUSTFLAGS:-} -L $CONDA_PREFIX/lib -C link-arg=-Wl,-rpath,$CONDA_PREFIX/lib"
if [ ! -d "$SCX_DIR/.git" ]; then git clone https://github.com/sched-ext/scx "$SCX_DIR"; fi
git -C "$SCX_DIR" fetch --tags origin
git -C "$SCX_DIR" checkout "$SCX_REF"
pargs=(); for s in $SCHEDS; do pargs+=(-p "$s"); done
wh_log "building ${SCHEDS} @ $(git -C "$SCX_DIR" rev-parse --short HEAD) (log: $SCX_DIR/build.log)"
( cd "$SCX_DIR" && wh_heavy cargo build --release -j1 "${pargs[@]}" ) 2>&1 | tee "$SCX_DIR/build.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || wh_die "cargo build failed (see $SCX_DIR/build.log)"
git -C "$SCX_DIR" rev-parse HEAD > "$SCX_DIR/WINHINT_SCX_COMMIT"
for s in $SCHEDS; do
  "$SCX_DIR/target/release/$s" --help > "$SCX_DIR/$s.help.txt" 2>&1 || true
done
wh_log "built scx @ $(cat "$SCX_DIR/WINHINT_SCX_COMMIT"); check the preset flags in run_scx.sh against $SCX_DIR/scx_*.help.txt"
