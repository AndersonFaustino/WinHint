# Sourced by the compiler test scripts. Enters the `winhint` micromamba env
# (re-execs the calling script under `micromamba run` if needed) and defines
# the common variables.
if [ -z "${CONDA_PREFIX:-}" ] || ! command -v llvm-config >/dev/null 2>&1; then
  MM=${MAMBA_EXE:-$HOME/.local/bin/micromamba}
  [ -x "$MM" ] || { echo "micromamba env 'winhint' not available" >&2; exit 1; }
  exec "$MM" run -n winhint bash "$0" "$@"
fi
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WINHINT_ROOT=${WINHINT_ROOT:-$(cd "$_here/../.." && pwd)}
WINHINT_BUILD=${WINHINT_BUILD:-$WINHINT_ROOT/build}
B=${COMPILER_BUILD:-$WINHINT_BUILD/compiler}
WH=$B/WinHint.so
JQ=$B/JonesIQ.so
MACHINE=${MACHINE:-$WINHINT_ROOT/sim/machines/riscv_ooo.json}
# RISC-V: clang 23 with the conda sysroot (build/ENV.md); x86-64: native.
RV_CFLAGS=${RV_CFLAGS:-${WINHINT_RISCV_CLANG_FLAGS:---target=riscv64-conda-linux-gnu --sysroot=$CONDA_PREFIX/riscv64-conda-linux-gnu/sysroot --gcc-toolchain=$CONDA_PREFIX -fuse-ld=lld} -march=rv64gc -mabi=lp64d}
X86_CFLAGS=${X86_CFLAGS:---target=x86_64-conda-linux-gnu}
WHF="-fplugin=$WH -fpass-plugin=$WH -mllvm -winhint-target=$MACHINE"
[ -f "$WH" ] || { echo "missing $WH: run compiler/build.sh" >&2; exit 1; }
