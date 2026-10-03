#!/usr/bin/env bash
# Build the WinHint LLVM pass plugins on the host, from the `winhint`
# micromamba env (docs/interfaces.md §1; no Docker).
#   compiler/build.sh                 # -> build/compiler/WinHint.so (+ JonesIQ.so)
# Env overrides: WINHINT_ROOT, WINHINT_BUILD, COMPILER_BUILD (default $WINHINT_BUILD/compiler;
#                conda sets $BUILD to a triple, so BUILD is not used),
#                JOBS (default 1: 6 GB RAM shared with other jobs), CXX,
#                WINHINT_COVERAGE=1 (gcov-instrumented -O0 build for tooling/coverage.sh;
#                use it with a separate COMPILER_BUILD).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WINHINT_ROOT=${WINHINT_ROOT:-$(cd "$HERE/.." && pwd)}
WINHINT_BUILD=${WINHINT_BUILD:-$WINHINT_ROOT/build}
BUILD=${COMPILER_BUILD:-$WINHINT_BUILD/compiler}
JOBS=${JOBS:-1}

# Use the env when we are not already inside it.
if ! command -v llvm-config >/dev/null 2>&1 || [ -z "${CONDA_PREFIX:-}" ]; then
  MM=${MAMBA_EXE:-$HOME/.local/bin/micromamba}
  if [ -x "$MM" ]; then
    exec "$MM" run -n winhint bash "$0" "$@"
  fi
  echo "[ERROR] llvm-config not found and micromamba env 'winhint' unavailable (tooling/create_conda_env.sh)" >&2
  exit 1
fi

CXX=${CXX:-$(command -v clang++)}
# Coverage: no build type (Release would append -O3), -DNDEBUG kept so the plugin matches the
# release LLVM headers (ABI-breaking checks).
COV_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if [ "${WINHINT_COVERAGE:-0}" = 1 ]; then
  COV_ARGS=(-DCMAKE_BUILD_TYPE= -DCMAKE_CXX_FLAGS="--coverage -O0 -g -DNDEBUG"
            -DCMAKE_MODULE_LINKER_FLAGS=--coverage -DCMAKE_SHARED_LINKER_FLAGS=--coverage)
fi
cmake -G Ninja -S "$HERE" -B "$BUILD" \
  "${COV_ARGS[@]}" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DLLVM_DIR="$(llvm-config --cmakedir)" >/dev/null
ninja -C "$BUILD" -j"$JOBS"
ls -l "$BUILD"/*.so
