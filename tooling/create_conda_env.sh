#!/usr/bin/env bash
#
# Create a self-contained conda environment for WinHint: Python, every tool
# WinHint drives (the LLVM 23 / GCC 13 host toolchain, the RISC-V cross GCC and
# user-mode QEMU, the gem5 and McPAT build dependencies, Rust, perf, Doxygen,
# gcovr), from conda-forge, and its Python dependencies (requirements*.txt).
#
#   tooling/create_conda_env.sh                        # env "winhint", dev + docs tools
#   tooling/create_conda_env.sh --extras dev           # without the documentation tools
#   tooling/create_conda_env.sh --update               # sync an existing env with the files
#   tooling/create_conda_env.sh --verify-only          # only run the checks
#   tooling/create_conda_env.sh --dry-run              # print the commands only
#   tooling/create_conda_env.sh --help                 # every option
#
# environment.yml is the declarative equivalent of the toolchain table below
# (`conda env create -f environment.yml`); the pins also live in
# tooling/versions.lock, which the verification checks the installed env against.
set -euo pipefail

# --------------------------------------------------------------------------
# WinHint's part: the toolchain groups it needs beyond base and compilers,
# and how its Python side is installed and checked.
# --------------------------------------------------------------------------
PROJECT="WinHint"
ENV_NAME="winhint"
EXTRAS="dev,docs"         # `make validate` builds the docs, so the gate needs both
TOOL_GROUPS=(riscv gem5 hw docs coverage)
USES_TORCH=1
PERF_PARANOID_MAX=1
PERF_FROM_ENV=1           # conda-forge linux-perf, verified with the hw group
PERF_NOTE="the real-hardware campaign (hw/)"
BUILD_ON_ACTIVATE=0       # the activation hook below sets WinHint's own variables
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WINHINT_BUILD="${WINHINT_BUILD:-$REPO_ROOT/build}"
LOCK_FILE="${WINHINT_VERSIONS_LOCK:-$REPO_ROOT/tooling/versions.lock}"   # override only for testing
ENV38_NAME="winhint-llvm38"
ENVK_NAME="winhint-kmod"
PROBE_COMPILERS=(x86_64-conda-linux-gnu-gcc:c x86_64-conda-linux-gnu-g++:cc)
MODE="create"             # create | update | verify
DO_LLVM38=1
DO_KMOD=1

project_usage() {
    cat <<'EOF'

Extras: dev (pytest, pytest-cov, coverage, pre-commit), docs (mkdocs, Material,
mkdocstrings, mkdoxy), optional (none: WinHint has no optional Python
packages). The commit gate also builds the docs, so contributors want dev,docs.

WinHint options:
      --update          Sync the existing environment with environment.yml's
                        toolchain table and requirements*.txt (no --force).
      --verify-only     Only run the checks, against tooling/versions.lock.
      --skip-llvm38     Do not create winhint-llvm38 (LLVM 3.8.1, baseline B8).
      --skip-kmod       Do not create winhint-kmod (GCC 15.2, PMCTrack module).
Both extra environments stay separate: they pin compilers that conflict with
the LLVM 23 / GCC 13 of the main one.
EOF
}

project_option() {
    case "$1" in
        --update)      MODE=update;  SHIFT=1 ;;
        --verify-only) MODE=verify;  SHIFT=1 ;;
        --skip-llvm38) DO_LLVM38=0;  SHIFT=1 ;;
        --skip-kmod)   DO_KMOD=0;    SHIFT=1 ;;
        *)             return 1 ;;
    esac
}

lock() { grep -E "^$1=" "$LOCK_FILE" | head -1 | cut -d= -f2- | tr -d '[:space:]'; }

# The env's own tools, from outside it.
renv() { "$CONDA" run -n "$ENV_NAME" "$@"; }

other_env_exists() { [ -d "$(dirname "$(dirname "$PREFIX")")/envs/$1/conda-meta" ]; }

project_install() {
    local PIP=(python -m pip install)
    # PyTorch first, built for the accelerator, so pip keeps it when
    # requirements.txt asks for torch. An update keeps the one in place.
    if [ "$MODE" != update ] || ! renv python -c 'import torch' >/dev/null 2>&1; then
        install_torch
    fi
    info "installing the runtime dependencies"
    in_env "${PIP[@]}" -r "$REPO_ROOT/requirements.txt"
    if want dev; then
        info "installing the test and commit-gate tools"
        in_env "${PIP[@]}" -r "$REPO_ROOT/requirements-dev.txt"
    fi
    if want docs; then
        info "installing the documentation tools"
        in_env "${PIP[@]}" -r "$REPO_ROOT/requirements-docs.txt"
    fi
    if want optional; then
        info "installing the optional dependencies (none for WinHint)"
        in_env "${PIP[@]}" -r "$REPO_ROOT/requirements-optional.txt"
    fi
    write_winhint_activation
    link_qemu
    if [ "$DO_LLVM38" -eq 1 ]; then create_llvm38_env; fi
    if [ "$DO_KMOD" -eq 1 ]; then create_kmod_env; fi
}

# WINHINT_ROOT / WINHINT_BUILD, PATH and the RISC-V cross variables, on top of
# the toolchain.sh activation script written by the shared part.
write_winhint_activation() {
    info "writing the WinHint activation hook"
    [ "$DRY_RUN" -eq 1 ] && return 0
    mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d" "$WINHINT_BUILD/bin"
    cat > "$PREFIX/etc/conda/activate.d/zz_winhint.sh" <<HOOK
# Written by WinHint's tooling/create_conda_env.sh.
export WINHINT_ROOT="\${WINHINT_ROOT:-$REPO_ROOT}"
export WINHINT_BUILD="\${WINHINT_BUILD:-\${WINHINT_ROOT}/build}"
export _WINHINT_OLD_PATH="\${PATH}"
export PATH="\${WINHINT_ROOT}/tooling:\${WINHINT_BUILD}/bin:\${CONDA_PREFIX}/libexec/llvm:\${PATH}"
# RISC-V cross toolchain (conda-forge gcc_impl_linux-riscv64)
export WINHINT_RISCV_CC="riscv64-conda-linux-gnu-gcc"
export WINHINT_RISCV_CXX="riscv64-conda-linux-gnu-g++"
export WINHINT_RISCV_TRIPLE="riscv64-conda-linux-gnu"
export WINHINT_RISCV_SYSROOT="\${CONDA_PREFIX}/riscv64-conda-linux-gnu/sysroot"
# Clang cross flags (the conda triple, so clang finds crt*/libgcc):
#   clang \$WINHINT_RISCV_CLANG_FLAGS -O2 foo.c -o foo
export WINHINT_RISCV_CLANG_FLAGS="--target=riscv64-conda-linux-gnu --sysroot=\${WINHINT_RISCV_SYSROOT} --gcc-toolchain=\${CONDA_PREFIX} -fuse-ld=lld"
export QEMU_LD_PREFIX="\${WINHINT_RISCV_SYSROOT}"
HOOK
    cat > "$PREFIX/etc/conda/deactivate.d/zz_winhint.sh" <<'HOOK'
# Written by WinHint's tooling/create_conda_env.sh.
if [ -n "${_WINHINT_OLD_PATH:-}" ]; then export PATH="${_WINHINT_OLD_PATH}"; fi
unset _WINHINT_OLD_PATH WINHINT_ROOT WINHINT_BUILD WINHINT_RISCV_CC WINHINT_RISCV_CXX WINHINT_RISCV_TRIPLE WINHINT_RISCV_SYSROOT WINHINT_RISCV_CLANG_FLAGS QEMU_LD_PREFIX
HOOK
}

# conda-forge's qemu-execve-riscv64 may only ship a prefixed binary; expose the
# standard name.
link_qemu() {
    local q
    [ "$DRY_RUN" -eq 1 ] && return 0
    [ -x "$PREFIX/bin/qemu-riscv64" ] && return 0
    q="$(ls "$PREFIX"/bin/*qemu*riscv64* 2>/dev/null | head -1 || true)"
    [ -n "$q" ] || die "no qemu riscv64 binary in $PREFIX/bin (package qemu-execve-riscv64)"
    ln -sf "$(basename "$q")" "$PREFIX/bin/qemu-riscv64"
}

# Clairvoyance (B8). conda-forge ships llvmdev/clangdev 3.8.1 built for old
# conda; their `system` dependency is not on modern channels, so numba's
# channel is searched too (see versions.lock).
create_llvm38_env() {
    local v38; v38="$(lock LLVM38)"
    if other_env_exists "$ENV38_NAME"; then
        [ "$FORCE" -eq 1 ] || { info "$ENV38_NAME exists"; return 0; }
        run "$CONDA" env remove -y -n "$ENV38_NAME"
    fi
    info "creating $ENV38_NAME (LLVM $v38)"
    run "$CONDA" create -n "$ENV38_NAME" -y -c conda-forge -c numba "llvmdev=$v38" "clangdev=$v38" cmake make \
        || die "could not create $ENV38_NAME (no LLVM $v38 conda package?); B8 goes straight to the port (docs/reference/proposal.md)"
}

# The PMCTrack kernel module (R4) must be built with the host kernel's compiler.
create_kmod_env() {
    local vk; vk="$(lock KMOD_GCC)"
    if other_env_exists "$ENVK_NAME"; then
        [ "$FORCE" -eq 1 ] || { info "$ENVK_NAME exists"; return 0; }
        run "$CONDA" env remove -y -n "$ENVK_NAME"
    fi
    info "creating $ENVK_NAME (GCC $vk)"
    run "$CONDA" create -n "$ENVK_NAME" -y -c conda-forge "gcc_linux-64=$vk" make \
        || die "could not create $ENVK_NAME (gcc_linux-64=$vk)"
}

# The installed versions against tooling/versions.lock, a RISC-V hello world
# under QEMU, the extra environments, and the one-screen summary build/ENV.md.
project_verify() {
    local rc=0 got_py got_llvm got_gcc got_rv got_qemu got_sys got_glib tdir want

    verify_version() {   # what, got, want, glob-suffix
        if [ -z "${3:-}" ] || [[ "$2" == "$3"$4 ]]; then
            printf '   ok       %-12s %s\n' "$1" "$2"
        else
            printf '   broken   %-12s %s != versions.lock %s\n' "$1" "${2:-missing}" "$3"; rc=1
        fi
    }

    if renv python -c 'import sys, torch, pandas, numpy, sklearn' >/dev/null 2>&1; then
        printf '   ok       %-12s torch, pandas, numpy, sklearn import\n' python
    else
        printf '   broken   %-12s torch/pandas/numpy/sklearn do not import\n' python; rc=1
    fi
    got_py="$(renv python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    got_llvm="$(renv llvm-config --version)"; got_llvm="${got_llvm%git}"
    got_gcc="$(renv bash -c '"$CXX" -dumpfullversion')"
    got_rv="$(renv riscv64-conda-linux-gnu-gcc -dumpfullversion)"
    got_qemu="$(renv qemu-riscv64 --version | grep -m1 -oE 'version [0-9][0-9.]*' | cut -d' ' -f2)"
    got_sys="$(ls -d "$PREFIX"/conda-meta/sysroot_linux-riscv64-[0-9]*.json 2>/dev/null | head -1 | sed -E 's/.*sysroot_linux-riscv64-([0-9.]+)-.*/\1/')"
    got_glib="$(ls -d "$PREFIX"/conda-meta/glib-[0-9]*.json 2>/dev/null | head -1 | sed -E 's/.*glib-([0-9.]+)-.*/\1/')"
    verify_version python "$got_py" "$(lock PYTHON)" ""
    verify_version llvm "$got_llvm" "$(lock LLVM)" ""
    verify_version host-gcc "$got_gcc" "$(lock HOST_GCC)" "*"
    verify_version riscv-gcc "$got_rv" "$(lock RISCV_GCC)" "*"
    verify_version qemu "$got_qemu" "$(lock QEMU)" ""
    verify_version sysroot "$got_sys" "$(lock RISCV_SYSROOT)" "*"
    verify_version glib "$got_glib" "$(lock GLIB)" ""
    if want docs; then
        for want in "mkdocs:DOCS_MKDOCS" "mkdocs-material:DOCS_MATERIAL" \
                    "mkdocstrings-python:DOCS_MKDOCSTRINGS_PYTHON" "mkdoxy:DOCS_MKDOXY"; do
            verify_version "${want%%:*}" \
                "$(renv python -c "import importlib.metadata as m; print(m.version('${want%%:*}'))" 2>/dev/null)" \
                "$(lock "${want##*:}")" ""
        done
        verify_version doxygen "$(renv doxygen --version | cut -d' ' -f1)" "$(lock DOCS_DOXYGEN)" ""
    fi

    tdir="$(mktemp -d)"
    printf '#include <stdio.h>\nint main(void) { printf("hello riscv\\n"); return 0; }\n' > "$tdir/hello.c"
    for want in \
        "riscv-gcc-static:riscv64-conda-linux-gnu-gcc -O2 -static $tdir/hello.c -o $tdir/h1 && qemu-riscv64 $tdir/h1" \
        "riscv-gcc-dyn:riscv64-conda-linux-gnu-gcc -O2 $tdir/hello.c -o $tdir/h2 && qemu-riscv64 $tdir/h2" \
        "riscv-clang:clang \$WINHINT_RISCV_CLANG_FLAGS -O2 $tdir/hello.c -o $tdir/h3 && qemu-riscv64 $tdir/h3"; do
        if [ "$(renv bash -c "${want#*:}" 2>/dev/null)" = "hello riscv" ]; then
            printf '   ok       %-12s hello under qemu-riscv64\n' "${want%%:*}"
        else
            printf '   broken   %-12s cannot build and run hello under qemu-riscv64\n' "${want%%:*}"; rc=1
        fi
    done
    rm -rf "$tdir"

    if other_env_exists "$ENV38_NAME"; then
        verify_version llvm-3.8 "$("$CONDA" run -n "$ENV38_NAME" llvm-config --version 2>/dev/null)" "$(lock LLVM38)" ""
    fi
    if other_env_exists "$ENVK_NAME"; then
        verify_version kmod-gcc "$("$CONDA" run -n "$ENVK_NAME" x86_64-conda-linux-gnu-gcc -dumpfullversion 2>/dev/null)" "$(lock KMOD_GCC)" "*"
    fi

    write_env_summary "$got_gcc" "$got_llvm" "$got_rv" "$got_qemu"
    return "$rc"
}

# build/ENV.md, referenced by the component docs, and the .env-ready marker.
write_env_summary() {
    local got_gcc="$1" got_llvm="$2" got_rv="$3" got_qemu="$4" activate
    mkdir -p "$WINHINT_BUILD"
    if [ "$(basename "$CONDA")" = micromamba ]; then
        activate="eval \"\$($CONDA shell hook -s bash)\" && micromamba activate $ENV_NAME"
    else
        activate="conda activate $ENV_NAME"
    fi
    cat > "$WINHINT_BUILD/ENV.md" <<MD
# WinHint environment — READY (conda env \`$ENV_NAME\`, host, no Docker)
<!-- Written by tooling/create_conda_env.sh — do not edit. -->

Activate:   $activate
One-shot:   $CONDA run -n $ENV_NAME <cmd>
Activation sets: WINHINT_ROOT (repo), WINHINT_BUILD=\$WINHINT_ROOT/build, PATH += tooling/ build/bin/,
  CC/CXX = conda GCC $got_gcc, WINHINT_RISCV_{CC,CXX,TRIPLE,SYSROOT,CLANG_FLAGS}, QEMU_LD_PREFIX

| Tool | Command | Version |
|------|---------|---------|
| Python | python, pytest | $(renv python -c 'import sys,torch,pandas; print(sys.version.split()[0], "torch", torch.__version__, "pandas", pandas.__version__)' 2>/dev/null | head -1) |
| LLVM/Clang | clang, opt, llc, llvm-config, ld.lld; lit, FileCheck (libexec/llvm, on PATH) | $got_llvm |
| Host GCC | \$CC / \$CXX | $got_gcc |
| RISC-V GCC | riscv64-conda-linux-gnu-gcc / -g++ | $got_rv, sysroot $(lock RISCV_SYSROOT) |
| RISC-V clang | clang \$WINHINT_RISCV_CLANG_FLAGS (triple riscv64-conda-linux-gnu, not -unknown-) | $got_llvm |
| QEMU user | qemu-riscv64 (QEMU_LD_PREFIX = RISC-V sysroot) | $got_qemu |
| Build | scons, make, cmake, ninja, ccache, pkg-config, m4, git | scons $(renv bash -c 'scons --version | grep -m1 -o "v[0-9][0-9.]*[0-9]"' 2>/dev/null | head -1), cmake $(renv bash -c 'cmake --version | grep -o "[0-9][0-9.]*" | head -1' 2>/dev/null | head -1) |
| HW tools | rustc, cargo, perf | $(renv bash -c 'rustc --version | cut -d" " -f2' 2>/dev/null | head -1), perf $(renv bash -c 'perf --version | cut -d" " -f3' 2>/dev/null | head -1) |

Paths: docs/interfaces.md §1. Manager: tooling/winhint.sh help. Pins and substitutions: tooling/versions.lock.
Heavy jobs: flock "\$WINHINT_BUILD/.heavy.lock" <cmd>, max -j2; everything else -j1.
Documentation site: tooling/winhint.sh docs:build | docs:serve (MkDocs $(lock DOCS_MKDOCS) + Doxygen $(lock DOCS_DOXYGEN), in this env).
Other envs: $ENV38_NAME (LLVM $(lock LLVM38), B8), $ENVK_NAME (GCC $(lock KMOD_GCC) for kernel modules).
MD
    touch "$WINHINT_BUILD/.env-ready"
}

project_next() {
    cat <<'NEXT'

  tooling/winhint.sh status         # tools, builds, heavy lock
  make validate                     # the commit gate
  make install-hooks                # run it on every commit
  tooling/winhint.sh help           # every command
NEXT
}

# ============================================================================
# Everything below is the layout shared with the create_conda_env.sh of
# Jacucaca, Cambaxirra, Rolinha, COSY and Seriema: every environment gets its toolchain from
# one table, and
# PyTorch, where a project uses it, built for the host's accelerator. Only the
# project section above differs. It sets PROJECT, ENV_NAME, EXTRAS,
# EXTRA_NAMES (the --extras groups it knows, beyond dev, docs, optional, all
# and none), TOOL_GROUPS (the toolchain groups beyond base and compilers),
# USES_TORCH, PERF_PARANOID_MAX, PERF_NOTE and BUILD_ON_ACTIVATE, and defines
# the hooks project_usage, project_option (tried before the shared options),
# project_install (which calls install_torch when it wants PyTorch),
# project_verify and project_next.
# ============================================================================

# ---- the toolchain: exact conda-forge versions, one group per purpose -----
# perf is the one host tool: it belongs to the running kernel.
# WinHint keeps its own pins (tooling/versions.lock) instead of the versions the
# other projects share: gem5 needs GCC 13 and the LLVM passes are written for 23.
LLVM_VERSION="23.1.2"
PYTHON_DEFAULT="3.14"
declare -A TOOLCHAIN TOOLS PKGCONFIG
TOOLCHAIN[base]="git curl make cmake ninja pkg-config ccache m4"
TOOLS[base]="git curl make cmake ninja pkg-config ccache m4"
TOOLCHAIN[compilers]="gcc_linux-64=13 gxx_linux-64=13 clang=$LLVM_VERSION clangxx=$LLVM_VERSION clang-tools=$LLVM_VERSION llvmdev=$LLVM_VERSION llvm-tools=$LLVM_VERSION lld=$LLVM_VERSION compiler-rt=$LLVM_VERSION lit=$LLVM_VERSION compiler-rt_linux-riscv64=$LLVM_VERSION"
TOOLS[compilers]="x86_64-conda-linux-gnu-gcc x86_64-conda-linux-gnu-g++ clang clang++ llvm-config opt llc ld.lld lit FileCheck"
# conda-forge ships the RISC-V cross GCC only from 15.2 on, and the gcc_linux-riscv64
# wrapper pins gcc_impl_linux-64 to the same version (conflicts with host GCC 13),
# so the *_impl packages are used directly.
TOOLCHAIN[riscv]="gcc_impl_linux-riscv64=15.3 gxx_impl_linux-riscv64=15.3 sysroot_linux-riscv64=2.39 qemu-execve-riscv64"
TOOLS[riscv]="riscv64-conda-linux-gnu-gcc riscv64-conda-linux-gnu-g++ qemu-riscv64"
# gem5 / McPAT build dependencies
TOOLCHAIN[gem5]="scons zlib protobuf gperftools libpng libboost-devel hdf5"
TOOLS[gem5]="scons"
# real-hardware tools, and the R3 intel-lpmd build dependencies (upower-glib and
# libbpf/bpftool have no conda package: see versions.lock)
TOOLCHAIN[hw]="rust linux-perf libnl libsystemd libxml2-devel libglib=2.90.0 glib=2.90.0 glib-tools=2.90.0"
TOOLS[hw]="rustc cargo perf"
TOOLCHAIN[llvmdev]="llvmdev=$LLVM_VERSION clangdev=$LLVM_VERSION cmake=4.4.3 ninja=1.13.2 capstone=5.0.9 scikit-build-core=1.1.0 pybind11=3.1.0"
TOOLS[llvmdev]="llvm-config cmake ninja"
PKGCONFIG[llvmdev]="capstone"
# BOLT is not on conda-forge: install_bolt takes it from the LLVM release of
# the same version, and patchelf points it at the environment's libraries.
TOOLCHAIN[bolt]="patchelf=0.19.1 libzlib=1.3.2"
TOOLS[bolt]="llvm-bolt perf2bolt merge-fdata"
# CFGgrind is a Valgrind tool built into its own Valgrind: build_cfggrind.
TOOLCHAIN[cfggrind]="autoconf=2.72 automake=1.19 libtool=2.5.4 m4=1.4.21 patch=2.8"
TOOLS[cfggrind]="valgrind cfggrind_info"
TOOLCHAIN[docs]="doxygen=1.18.0"   # mkdoxy runs it
TOOLS[docs]="doxygen"
TOOLCHAIN[coverage]="gcovr"
TOOLS[coverage]="gcovr"
VALGRIND_VERSION="3.26.0"
CFGGRIND_COMMIT="e2cdbfbf8c79ec7238e0c2f31e8c64ea00f78a49"
BOLT_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/LLVM-$LLVM_VERSION-Linux-X64.tar.xz"
BOLT_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/conda-toolchain/bolt-$LLVM_VERSION"
MICROMAMBA_URL="https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"

# PyTorch CUDA indices, newest first, each with the minimum driver CUDA
# version that can run it. install_torch picks the newest the driver supports
# and walks down this list if an index turns out not to exist, so a stale
# table costs a retry rather than a failure.
CUDA_TABLE=(
    "cu130 13.0"
    "cu128 12.8"
    "cu126 12.6"
    "cu124 12.4"
    "cu121 12.1"
    "cu118 11.8"
)
ROCM_DEFAULT="rocm6.2"

PYTHON_VERSION=""
ACCEL="auto"
CUDA_TAG=""
ROCM_TAG=""
TORCH_ACCEL=""
JOBS=2
FORCE=0
DRY_RUN=0

die()  { printf 'error: %s\n' "$1" >&2; exit 1; }
info() { printf '\n==> %s\n' "$1"; }
warn() { printf '   warning: %s\n' "$1" >&2; }

usage() {
    cat <<EOF
Create the conda environment for $PROJECT, self-contained: Python, every
tool and toolchain it runs or builds with (from conda-forge, at the versions
pinned in tooling/versions.lock), and $PROJECT's Python dependencies.

Usage: tooling/create_conda_env.sh [options]

Options:
  -n, --name NAME       Environment name (default: $ENV_NAME).
  -p, --python VERSION  Python version (default: $PYTHON_DEFAULT).
  -e, --extras LIST     Comma-separated Python requirement groups:
                        dev, docs, optional, all, ${EXTRA_NAMES:+${EXTRA_NAMES// /, }, }or none
                        (default: $EXTRAS).
  -j, --jobs N          Parallel compile jobs for the source builds
                        (default: 2; an LLVM/Clang compile needs about a
                        gigabyte of memory).
  -f, --force           Remove an existing environment of the same name.
      --dry-run         Print the commands without running them.
  -h, --help            Show this help.
$(torch_usage; project_usage)

Toolchain groups installed: $(echo base compilers "${TOOL_GROUPS[@]}")
$(toolchain_listing)

With no mamba, conda or micromamba on PATH, micromamba is downloaded into
\$MAMBA_ROOT_PREFIX (default: ~/.local/share/mamba). The script never runs
\`conda init\` and never edits your shell profile.
EOF
}

# The PyTorch options, for the projects that install PyTorch.
torch_usage() {
    [ "$USES_TORCH" -eq 1 ] || return 0
    cat <<'EOF'
      --accel MODE      auto | cuda | rocm | cpu: the PyTorch build to
                        install (default: auto, from the host's GPU driver)
      --cuda-tag TAG    force a PyTorch CUDA index, e.g. cu128
      --rocm-tag TAG    force a PyTorch ROCm index, e.g. rocm6.2
EOF
}

# The packages of every installed group, and where the others come from.
toolchain_listing() {
    local group
    for group in base compilers "${TOOL_GROUPS[@]}"; do
        printf '  %-10s %s\n' "$group" "${TOOLCHAIN[$group]}"
        case "$group" in
            bolt)     printf '  %-10s llvm-bolt, perf2bolt, merge-fdata from the LLVM %s release (not on conda-forge)\n' "" "$LLVM_VERSION" ;;
            cfggrind) printf '  %-10s Valgrind %s with CFGgrind %s, built from source\n' "" "$VALGRIND_VERSION" "${CFGGRIND_COMMIT:0:12}" ;;
        esac
    done
}

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '   [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

parse_args() {
    while [ $# -gt 0 ]; do
        SHIFT=0
        if project_option "$@"; then
            shift "$SHIFT"
            continue
        fi
        case "$1" in
            -n|--name)   [ $# -ge 2 ] || die "--name needs a value"; ENV_NAME="$2"; shift 2 ;;
            -p|--python) [ $# -ge 2 ] || die "--python needs a value"; PYTHON_VERSION="$2"; shift 2 ;;
            -e|--extras) [ $# -ge 2 ] || die "--extras needs a value"; EXTRAS="$2"; shift 2 ;;
            -j|--jobs)   [ $# -ge 2 ] || die "--jobs needs a value"; JOBS="$2"; shift 2 ;;
            -f|--force)  FORCE=1; shift ;;
            --dry-run)   DRY_RUN=1; shift ;;
            -h|--help)   usage; exit 0 ;;
            --accel|--cuda-tag|--rocm-tag)
                         [ "$USES_TORCH" -eq 1 ] || die "$1: $PROJECT does not use PyTorch"
                         [ $# -ge 2 ] || die "$1 needs a value"
                         case "$1" in
                             --accel)    ACCEL="$2" ;;
                             --cuda-tag) CUDA_TAG="$2" ;;
                             --rocm-tag) ROCM_TAG="$2" ;;
                         esac
                         shift 2 ;;
            *)           die "unknown option '$1' (try --help)" ;;
        esac
    done
    [ -n "$PYTHON_VERSION" ] || PYTHON_VERSION="$PYTHON_DEFAULT"
    case "$ACCEL" in
        auto|cuda|rocm|cpu) ;;
        *) die "--accel must be auto, cuda, rocm or cpu (got '$ACCEL')" ;;
    esac
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs takes a positive number, got '$JOBS'"
    local group
    for group in $(printf '%s' "$EXTRAS" | tr ',' ' '); do
        [[ " dev docs optional all none ${EXTRA_NAMES:-} " == *" $group "* ]] \
            || die "--extras: unknown group '$group' (known: dev, docs, optional, all, ${EXTRA_NAMES:+${EXTRA_NAMES// /, }, }none)"
    done
}

want() { case ",$EXTRAS," in *",all,"*) return 0 ;; *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# Nothing from the host leaks into the environment: not the user
# site-packages (~/.local/lib/pythonX.Y, which pip would otherwise see, and
# uninstall from, when replacing a package), nor search paths pointing at
# host libraries and headers.
isolate_from_host() {
    export PYTHONNOUSERSITE=1
    unset PYTHONPATH PYTHONHOME PKG_CONFIG_PATH LD_LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
}

# ---- the conda front end --------------------------------------------------
find_conda() {
    local candidate
    CONDA=""
    for candidate in mamba conda micromamba; do
        if command -v "$candidate" >/dev/null 2>&1; then CONDA="$(command -v "$candidate")"; break; fi
    done
    if [ -z "$CONDA" ]; then
        export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
        CONDA="$MAMBA_ROOT_PREFIX/bin/micromamba"
        if [ ! -x "$CONDA" ]; then
            info "no conda front end on PATH: installing micromamba into $MAMBA_ROOT_PREFIX"
            run mkdir -p "$MAMBA_ROOT_PREFIX/bin"
            if command -v curl >/dev/null 2>&1; then
                run curl -fsSL -o "$CONDA" "$MICROMAMBA_URL"
            elif command -v wget >/dev/null 2>&1; then
                run wget -q -O "$CONDA" "$MICROMAMBA_URL"
            else
                die "neither curl nor wget is available to download micromamba; install Miniforge (https://github.com/conda-forge/miniforge) instead"
            fi
            run chmod +x "$CONDA"
        fi
    fi
    [ -x "$CONDA" ] && info "using $CONDA ($("$CONDA" --version 2>&1 | head -1))"
    # `conda run` buffers the output unless told not to; mamba 2 and
    # micromamba do not buffer, and do not take the flag.
    RUN_FLAGS=()
    if [ -x "$CONDA" ] && "$CONDA" run --help 2>&1 | grep -q -- '--no-capture-output'; then
        RUN_FLAGS=(--no-capture-output)
    fi
    return 0
}

in_env() { run "$CONDA" run "${RUN_FLAGS[@]}" -n "$ENV_NAME" "$@"; }

# The environment's directory, or nothing when it does not exist.
env_prefix() {
    [ -x "$CONDA" ] || return 0
    "$CONDA" env list --json 2>/dev/null | tr -d ' ",' | grep -E "/envs/${ENV_NAME}\$" | head -1 || true
}

# ---- the environment -------------------------------------------------------
create_env() {
    local group packages=()
    if [ -n "$(env_prefix)" ]; then
        [ "$FORCE" -eq 1 ] || die "environment '$ENV_NAME' already exists; pass --force to replace it, or --name to pick another"
        info "removing the existing environment '$ENV_NAME'"
        run "$CONDA" env remove -y -n "$ENV_NAME"
    fi
    for group in base compilers "${TOOL_GROUPS[@]}"; do
        read -r -a group_packages <<<"${TOOLCHAIN[$group]}"
        packages+=("${group_packages[@]}")
    done
    info "creating '$ENV_NAME': Python $PYTHON_VERSION and the toolchain ($(echo base compilers "${TOOL_GROUPS[@]}"))"
    run "$CONDA" create -n "$ENV_NAME" -y --override-channels -c conda-forge "python=$PYTHON_VERSION" pip "${packages[@]}"
    PREFIX="$(env_prefix)"
    if [ -z "$PREFIX" ]; then
        [ "$DRY_RUN" -eq 1 ] || die "created '$ENV_NAME' but cannot find its directory"
        PREFIX="<prefix of $ENV_NAME>"
    fi
    # Builds inside the environment use its compilers and find its LLVM/Clang
    # and libraries; the RPATH loads the environment's libraries whatever
    # LD_LIBRARY_PATH says.
    mapfile -t BUILD_ENV < <(build_vars "$PREFIX")
    BUILD_ENV=(env "${BUILD_ENV[@]}" "CMAKE_BUILD_PARALLEL_LEVEL=$JOBS")
}

# Sync an existing environment with the toolchain table (--update).
update_env() {
    local group packages=() group_packages
    PREFIX="$(env_prefix)"
    [ -n "$PREFIX" ] || [ "$DRY_RUN" -eq 1 ] || die "environment '$ENV_NAME' does not exist; run without --update"
    for group in base compilers "${TOOL_GROUPS[@]}"; do
        read -r -a group_packages <<<"${TOOLCHAIN[$group]}"
        packages+=("${group_packages[@]}")
    done
    info "updating '$ENV_NAME': Python $PYTHON_VERSION and the toolchain ($(echo base compilers "${TOOL_GROUPS[@]}"))"
    run "$CONDA" install -n "$ENV_NAME" -y --override-channels -c conda-forge "python=$PYTHON_VERSION" pip "${packages[@]}"
    mapfile -t BUILD_ENV < <(build_vars "$PREFIX")
    BUILD_ENV=(env "${BUILD_ENV[@]}" "CMAKE_BUILD_PARALLEL_LEVEL=$JOBS")
}

# NAME=VALUE for building inside an environment at prefix $1.
build_vars() {
    printf '%s\n' "CC=$1/bin/gcc" "CXX=$1/bin/g++" "CMAKE_PREFIX_PATH=$1" \
        "LLVM_DIR=$1/lib/cmake/llvm" "Clang_DIR=$1/lib/cmake/clang" \
        "CPPFLAGS=-I$1/include" "CFLAGS=-I$1/include" "CXXFLAGS=-I$1/include" \
        "LDFLAGS=-L$1/lib -Wl,-rpath,$1/lib"
}

# Activating the environment keeps the host's user site-packages out (and,
# with BUILD_ON_ACTIVATE=1, points builds at the environment's toolchain);
# deactivating restores what was there. conda, mamba and micromamba all
# source activate.d/deactivate.d.
write_activation() {
    local vars=("PYTHONNOUSERSITE=1") var name
    [ "${BUILD_ON_ACTIVATE:-0}" -eq 1 ] && mapfile -t -O 1 vars < <(build_vars '${CONDA_PREFIX}')
    info "isolating '$ENV_NAME' from the host on activation (${vars[*]%%=*})"
    [ "$DRY_RUN" -eq 1 ] && return 0
    mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
    {
        echo "# Written by $PROJECT's tooling/create_conda_env.sh."
        for var in "${vars[@]}"; do
            name="${var%%=*}"
            echo "export _TOOLCHAIN_OLD_$name=\"\${$name-__unset__}\""
            echo "export $name=\"${var#*=}\""
        done
    } > "$PREFIX/etc/conda/activate.d/toolchain.sh"
    {
        echo "# Written by $PROJECT's tooling/create_conda_env.sh."
        for var in "${vars[@]}"; do
            name="${var%%=*}"
            echo "if [ \"\${_TOOLCHAIN_OLD_$name-__unset__}\" = __unset__ ]; then unset $name; else export $name=\"\$_TOOLCHAIN_OLD_$name\"; fi"
            echo "unset _TOOLCHAIN_OLD_$name"
        done
    } > "$PREFIX/etc/conda/deactivate.d/toolchain.sh"
}

# ---- tools that conda-forge does not package ------------------------------
# BOLT from the LLVM release of the toolchain's version. Its tools link LLVM
# statically; their RPATH is set to the environment's lib/, so libstdc++,
# libgcc_s and zlib come from the environment too. Only BOLT is unpacked,
# once, into BOLT_CACHE (the release is ~1.8 GB), then copied into each
# environment.
install_bolt() {
    local file
    info "installing BOLT $LLVM_VERSION (llvm-bolt, perf2bolt, merge-fdata) from the LLVM release"
    if [ ! -x "$BOLT_CACHE/bin/llvm-bolt" ]; then
        run mkdir -p "$BOLT_CACHE.part"
        if [ "$DRY_RUN" -eq 1 ]; then
            printf '   [dry-run] curl -fsSL %s | tar -xJ -C %s (bin/llvm-bolt, bin/perf2bolt, bin/merge-fdata, lib/libbolt_rt*)\n' "$BOLT_URL" "$BOLT_CACHE"
        else
            "$CONDA" run -n "$ENV_NAME" curl -fsSL "$BOLT_URL" \
                | tar -xJ -C "$BOLT_CACHE.part" --strip-components=1 --wildcards --no-anchored \
                      'bin/llvm-bolt' 'bin/perf2bolt' 'bin/merge-fdata' 'lib/libbolt_rt*'
            rm -rf "$BOLT_CACHE" && mv "$BOLT_CACHE.part" "$BOLT_CACHE"
        fi
    fi
    [ "$DRY_RUN" -eq 1 ] && return 0
    for file in "$BOLT_CACHE"/bin/* "$BOLT_CACHE"/lib/*; do
        cp -P "$file" "$PREFIX/${file#"$BOLT_CACHE"/}"
    done
    for file in llvm-bolt merge-fdata; do
        "$CONDA" run -n "$ENV_NAME" patchelf --set-rpath '$ORIGIN/../lib' "$PREFIX/bin/$file"
    done
}

# CFGgrind (https://github.com/rimsa/CFGgrind) is a Valgrind tool: it is
# patched into Valgrind's source and built with the environment's compilers
# into the environment. As C17: CFGgrind names a field `bool`, a keyword in
# C23, which GCC 15 and later compile by default.
build_cfggrind() {
    info "building Valgrind $VALGRIND_VERSION with CFGgrind ${CFGGRIND_COMMIT:0:12} into '$ENV_NAME' ($JOBS jobs)"
    in_env bash -euc '
        work="$(mktemp -d)"; trap "rm -rf \"$work\"" EXIT
        cd "$work"
        curl -fsSL "https://sourceware.org/pub/valgrind/valgrind-$1.tar.bz2" | tar -xj
        cd "valgrind-$1"
        git init -q cfggrind && git -C cfggrind fetch -q --depth 1 https://github.com/rimsa/CFGgrind.git "$2" && git -C cfggrind checkout -q FETCH_HEAD
        patch -p1 -i cfggrind/cfggrind.patch >/dev/null
        ./autogen.sh >/dev/null
        CC="$CONDA_PREFIX/bin/gcc" CXX="$CONDA_PREFIX/bin/g++" CFLAGS="-O2 -g -std=gnu17" ./configure -q --prefix="$CONDA_PREFIX"
        make -s -j"$3"
        make -s install
    ' build_cfggrind "$VALGRIND_VERSION" "$CFGGRIND_COMMIT" "$JOBS"
}

# ---- PyTorch for the host's accelerator ------------------------------------
# The GPU driver, like perf, is the host's; the CUDA or ROCm runtime comes
# inside the PyTorch wheel of the matching index.

# `a >= b` for dotted version numbers.
version_ge() {
    [[ "$1" == "$2" ]] && return 0
    [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$2" ]]
}

# Presence of the hardware is not enough: without a working driver there is
# no CUDA to target, so the test is that nvidia-smi actually runs and lists a
# GPU. Only the chosen mode goes to stdout; the rest goes to stderr.
detect_accel() {
    local gpus
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        gpus="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
        if [[ "${gpus:-0}" -gt 0 ]]; then
            printf '   found %s NVIDIA GPU(s):\n' "$gpus" >&2
            nvidia-smi -L 2>/dev/null | sed 's/^/      /' >&2
            echo cuda
            return
        fi
    fi
    if { command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; } || [[ -d /opt/rocm ]]; then
        printf '   found an AMD ROCm installation\n' >&2
        echo rocm
        return
    fi
    if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qi 'nvidia'; then
        warn "an NVIDIA device is present but nvidia-smi does not run; treating this host as CPU-only (install the driver, or pass --accel cuda)"
    fi
    echo cpu
}

# Every PyTorch index to try for an accelerator, best first.
torch_indices() {
    local driver entry
    case "$1" in
        cuda)
            if [[ -n "$CUDA_TAG" ]]; then echo "$CUDA_TAG"; return; fi
            driver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)"
            [[ -z "$driver" ]] || printf '   the driver supports CUDA %s\n' "$driver" >&2
            for entry in "${CUDA_TABLE[@]}"; do
                if [[ -z "$driver" ]] || version_ge "$driver" "${entry##* }"; then echo "${entry%% *}"; fi
            done ;;
        rocm) echo "${ROCM_TAG:-$ROCM_DEFAULT}" ;;
        cpu)  echo cpu ;;
    esac
}

# Install PyTorch ($1: the requirement, default torch) built for the host's
# accelerator. Called before the requirement files, so pip keeps this build
# rather than pulling the default PyPI wheel over it.
install_torch() {
    local requirement="${1:-torch}" tag indices=()
    TORCH_ACCEL="$ACCEL"
    [[ "$TORCH_ACCEL" != auto ]] || TORCH_ACCEL="$(detect_accel)"
    mapfile -t indices < <(torch_indices "$TORCH_ACCEL")
    if [[ ${#indices[@]} -eq 0 ]]; then
        warn "the driver's CUDA is older than every PyTorch CUDA build this script knows about; installing the CPU build (update the driver for GPU support)"
        TORCH_ACCEL=cpu
        indices=(cpu)
    fi
    info "installing PyTorch for $TORCH_ACCEL (indices to try: ${indices[*]})"
    for tag in "${indices[@]}"; do
        if in_env python -m pip install --index-url "https://download.pytorch.org/whl/$tag" "$requirement"; then
            printf '   installed from index %s\n' "$tag"
            return 0
        fi
        warn "index '$tag' did not work; falling back to the next one"
    done
    die "could not install PyTorch from any index for '$TORCH_ACCEL'; pick one with --cuda-tag/--rocm-tag, or use --accel cpu"
}

# A GPU build that reports no device means the wheel and the driver disagree.
verify_torch() {
    [ -n "$TORCH_ACCEL" ] || return 0
    "$CONDA" run -n "$ENV_NAME" python -c '
import sys

import torch

gpu = torch.cuda.is_available()
print(f"   ok       torch        {torch.__version__} (built for {sys.argv[1]}; GPU available: {gpu}, devices: {torch.cuda.device_count()})")
sys.exit(sys.argv[1] in ("cuda", "rocm") and not gpu)
' "$TORCH_ACCEL" || {
        printf '   broken   torch        built for %s but sees no GPU: the wheel and the driver disagree (try a lower --cuda-tag, or --accel cpu)\n' "$TORCH_ACCEL"
        STATUS=1
    }
}

# ---- verification ----------------------------------------------------------
# Every tool of every group resolves inside the environment, the compilers
# build and run C and C++ programs with the environment's libraries, and perf
# is on the host.
verify_toolchain() {
    local group tool found module cc probe
    for group in base compilers "${TOOL_GROUPS[@]}"; do
        for tool in ${TOOLS[$group]}; do
            found="$("$CONDA" run -n "$ENV_NAME" bash -c "command -v '$tool'" 2>/dev/null || true)"
            case "$found" in
                "$PREFIX"/*) printf '   ok       %-12s %s\n' "$tool" "$found" ;;
                "")          printf '   missing  %-12s (%s)\n' "$tool" "$group"; STATUS=1 ;;
                *)           printf '   host     %-12s %s (not from the environment)\n' "$tool" "$found"; STATUS=1 ;;
            esac
        done
        for module in ${PKGCONFIG[$group]:-}; do
            if "$CONDA" run -n "$ENV_NAME" pkg-config --exists "$module" 2>/dev/null; then
                printf '   ok       %-12s pkg-config\n' "$module"
            else
                printf '   missing  %-12s pkg-config (%s)\n' "$module" "$group"; STATUS=1
            fi
        done
    done

    probe="$(mktemp -d)"
    printf '#include <stdio.h>\nint main(void) { puts("ok"); return 0; }\n' > "$probe/probe.c"
    printf '#include <iostream>\nint main() { std::cout << "ok" << std::endl; }\n' > "$probe/probe.cc"
    for cc in "${PROBE_COMPILERS[@]:-gcc:c clang:c g++:cc clang++:cc}"; do
        if "$CONDA" run -n "$ENV_NAME" "${cc%%:*}" -O2 "$probe/probe.${cc#*:}" -o "$probe/probe" >/dev/null 2>&1 \
           && [ "$(env -u LD_LIBRARY_PATH "$probe/probe")" = ok ]; then
            printf '   ok       %-12s builds and runs a program\n' "${cc%%:*}"
        else
            printf '   broken   %-12s cannot build and run a program\n' "${cc%%:*}"; STATUS=1
        fi
    done
    rm -rf "$probe"
}

verify_perf() {
    local paranoid
    if [ "${PERF_FROM_ENV:-0}" -eq 1 ]; then
        paranoid="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"
        [ "$paranoid" = "?" ] || [ "$paranoid" -le "$PERF_PARANOID_MAX" ] \
            || warn "kernel.perf_event_paranoid is $paranoid; $PERF_NOTE needs <= $PERF_PARANOID_MAX: sudo sysctl kernel.perf_event_paranoid=$PERF_PARANOID_MAX"
    elif command -v perf >/dev/null 2>&1 && perf --version >/dev/null 2>&1; then
        printf '   host     %-12s %s (%s)\n' perf "$(command -v perf)" "$(perf --version 2>&1)"
        paranoid="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"
        [ "$paranoid" = "?" ] || [ "$paranoid" -le "$PERF_PARANOID_MAX" ] \
            || warn "kernel.perf_event_paranoid is $paranoid; $PERF_NOTE needs <= $PERF_PARANOID_MAX: sudo sysctl kernel.perf_event_paranoid=$PERF_PARANOID_MAX"
    else
        warn "perf is not on the host (it belongs to the running kernel, so the environment does not carry it); $PERF_NOTE needs it (tooling/install_system_deps.sh perf)"
    fi
}

# ---- main -------------------------------------------------------------------
main() {
    local group
    parse_args "$@"
    isolate_from_host
    find_conda
    if [ "$MODE" = verify ]; then
        PREFIX="$(env_prefix)"
        [ -n "$PREFIX" ] || die "environment '$ENV_NAME' does not exist; run without --verify-only"
    else
        if [ "$MODE" = update ]; then update_env; else create_env; fi
        write_activation
        for group in "${TOOL_GROUPS[@]}"; do
            case "$group" in
                bolt)     install_bolt ;;
                cfggrind) build_cfggrind ;;
            esac
        done
        project_install
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        info "dry run: nothing was changed; after a real run:"
    else
        info "verifying '$ENV_NAME' ($PREFIX)"
        STATUS=0
        verify_toolchain
        project_verify || STATUS=1
        verify_torch
        verify_perf
        [ "$STATUS" -eq 0 ] || die "the environment is incomplete (see above)"
        info "done: $PREFIX"
    fi
    if [ "$(basename "$CONDA")" = micromamba ]; then
        printf '\n  eval "$(%s shell hook -s bash)" && micromamba activate %s\n' "$CONDA" "$ENV_NAME"
    else
        printf '\n  conda activate %s\n' "$ENV_NAME"
    fi
    project_next
}

main "$@"
