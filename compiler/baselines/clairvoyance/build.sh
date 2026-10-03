#!/usr/bin/env bash
# B8 Clairvoyance: build the artifact's LLVM 3.8 passes (third_party/clairvoyance,
# CGO'17) against the conda LLVM 3.8.1 of the micromamba env `winhint-llvm38`.
# Host only, no Docker; LLVM itself is NOT built (it comes from conda).
#
#   compiler/baselines/clairvoyance/build.sh          # -> $WINHINT_BUILD/clairvoyance/lib/*.so
#
# The C++ compiler is the `winhint` env's GCC 13 (x86_64-conda-linux-gnu-g++),
# in C++11 mode with the pre-C++11 libstdc++ ABI (the conda LLVM 3.8 build uses it),
# plus a few portability flags for the 3.8 headers. Runs under the heavy lock
# (docs/interfaces.md §1), -j1 by default.
#
# Env: WINHINT_ROOT, WINHINT_BUILD, JOBS (default 1), LLVM38_PREFIX
#      (default $MAMBA_ROOT_PREFIX/envs/winhint-llvm38), CV_CXX.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WINHINT_ROOT=${WINHINT_ROOT:-$(cd "$HERE/../../.." && pwd)}
WINHINT_BUILD=${WINHINT_BUILD:-$WINHINT_ROOT/build}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}
MM=${MAMBA_EXE:-$HOME/.local/bin/micromamba}
LLVM38_PREFIX=${LLVM38_PREFIX:-$MAMBA_ROOT_PREFIX/envs/winhint-llvm38}
JOBS=${JOBS:-1}
CV_SRC=$WINHINT_ROOT/third_party/clairvoyance/compiler
OUT=$WINHINT_BUILD/clairvoyance

# Run inside the winhint env (cmake, ninja, GCC 13) when not already there.
if [ "${CONDA_DEFAULT_ENV:-}" != winhint ]; then
  [ -x "$MM" ] || { echo "[ERROR] micromamba not found (tooling/create_conda_env.sh)" >&2; exit 1; }
  exec "$MM" run -n winhint bash "$0" "$@"
fi
[ -x "$LLVM38_PREFIX/bin/llvm-config" ] || {
  echo "[ERROR] $LLVM38_PREFIX has no LLVM 3.8 (tooling/create_conda_env.sh creates winhint-llvm38)." >&2
  echo "        Without LLVM 3.8 the passes must be ported to LLVM 23 (README.md, Port plan)." >&2
  exit 1; }
[ -d "$CV_SRC/projects" ] || {
  echo "[ERROR] $CV_SRC missing: git submodule update --init third_party/clairvoyance" >&2; exit 1; }
[ "$("$LLVM38_PREFIX/bin/llvm-config" --version)" = 3.8.1 ] || echo "[WARN] LLVM38 is not 3.8.1" >&2

CXX_HOST=${CV_CXX:-$(command -v x86_64-conda-linux-gnu-g++ || command -v g++)}
# 3.8 headers vs GCC 13: old ABI, C++11, missing transitive includes, no -Werror.
COMPAT="-std=gnu++11 -D_GLIBCXX_USE_CXX11_ABI=0 -include cstdint -include limits -include string \
-include cstdio -include algorithm -include memory -w"

mkdir -p "$OUT"
# Header shims (the conda package is not modified): GCC >= 6 rejects the implicit
# unique_ptr -> bool conversion in 3.8's ValueMap.h. Patched copies go in front of
# the 3.8 include dir.
SHIM=$OUT/shim-include
mkdir -p "$SHIM/llvm/IR"
sed 's/bool hasMD() const { return MDMap; }/bool hasMD() const { return bool(MDMap); }/' \
  "$LLVM38_PREFIX/include/llvm/IR/ValueMap.h" > "$SHIM/llvm/IR/ValueMap.h"
COMPAT="-I$SHIM $COMPAT"
# Patched copy of the artifact's pass sources (the submodule stays pristine):
# patches/*.patch fix undefined behaviour that modern GCC miscompiles.
SRC_COPY=$OUT/src
STAMP=$SRC_COPY/.patched
PATCHES=("$HERE"/patches/*.patch)
if [ ! -f "$STAMP" ] || [ -n "$(find "$CV_SRC/projects" "${PATCHES[@]}" -newer "$STAMP" -print -quit)" ]; then
  rm -rf "$SRC_COPY"; mkdir -p "$SRC_COPY"
  cp -a "$CV_SRC/projects" "$SRC_COPY/"
  for p in "${PATCHES[@]}"; do
    patch -s -p1 -d "$SRC_COPY" < "$p" || { echo "[ERROR] patch $p failed" >&2; exit 1; }
  done
  touch "$STAMP"
fi
COMPAT="$COMPAT -Werror=return-type"
LOCK=$WINHINT_BUILD/.heavy.lock
run_locked() { if [ -n "${WINHINT_NO_LOCK:-}" ]; then "$@"; else flock "$LOCK" "$@"; fi; }

if [ ! -f "$OUT/cmake/build.ninja" ]; then
  # Unset the conda compiler flags (-march=nocona -fno-plt ... -isystem $CONDA_PREFIX/include
  # would pull LLVM 23 headers in front of the 3.8 ones).
  env -u CXXFLAGS -u CPPFLAGS -u CFLAGS -u LDFLAGS -u CMAKE_PREFIX_PATH \
  cmake -G Ninja -S "$HERE/cmake" -B "$OUT/cmake" -DCV_SRC="$SRC_COPY" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Wno-deprecated -Wno-author \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$LLVM38_PREFIX/share/llvm/cmake" \
    -DCMAKE_CXX_COMPILER="$CXX_HOST" -DCMAKE_C_COMPILER="${CXX_HOST%++}cc" \
    -DCMAKE_CXX_FLAGS="$COMPAT" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    -DCMAKE_MODULE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"
fi
unset CXXFLAGS CPPFLAGS CFLAGS LDFLAGS
run_locked ninja -C "$OUT/cmake" -j"$JOBS"
mkdir -p "$OUT/lib"
cp -f "$OUT"/cmake/lib/*.so "$OUT/lib/"
ls -l "$OUT"/lib/*.so
# Smoke test: every pass library loads into opt 3.8.
for so in "$OUT"/lib/*.so; do
  "$LLVM38_PREFIX/bin/opt" -load "$so" -help >/dev/null 2>&1 \
    || { echo "[ERROR] opt 3.8 cannot load $so" >&2; exit 1; }
done
echo "[OK] Clairvoyance passes in $OUT/lib (LLVM $("$LLVM38_PREFIX/bin/llvm-config" --version))"
