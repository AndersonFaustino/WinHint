#!/usr/bin/env bash
# =============================================================================
# tooling/winhint.sh — WinHint project manager (host only, no Docker).
#
# Everything runs inside the micromamba env `winhint` (tooling/create_conda_env.sh);
# the script re-executes itself through `micromamba run -n winhint` when it is
# not already inside the env. Everything fetched or built goes under
# $WINHINT_BUILD (default <repo>/build). Run `tooling/winhint.sh help`.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WINHINT_ROOT="${WINHINT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export WINHINT_BUILD="${WINHINT_BUILD:-${WINHINT_ROOT}/build}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.local/share/mamba}"
MM="${MICROMAMBA:-$(command -v micromamba || echo "${MAMBA_ROOT_PREFIX}/bin/micromamba")}"
LOCK_FILE="${SCRIPT_DIR}/versions.lock"

GEM5_ROOT="${WINHINT_BUILD}/gem5"
GEM5_SRC="${GEM5_ROOT}/src"
MCPAT_ROOT="${WINHINT_BUILD}/mcpat"
MCPAT_SRC="${MCPAT_ROOT}/src"
HEAVY_LOCK="${WINHINT_BUILD}/.heavy.lock"
JOBS="${JOBS:-2}"                       # heavy builds: at most -j2 (6 GB RAM)
(( JOBS > 2 )) && JOBS=2

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}==>${NC} $*"; }
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }
lock() { grep -E "^$1=" "${LOCK_FILE}" | head -1 | cut -d= -f2- | tr -d '[:space:]'; }

in_env() { [[ "${CONDA_PREFIX:-}" == "${MAMBA_ROOT_PREFIX}/envs/winhint" ]]; }
ensure_env() {
    in_env && return 0
    [[ -d "${MAMBA_ROOT_PREFIX}/envs/winhint/conda-meta" ]] \
        || die "micromamba env 'winhint' not found — run: tooling/winhint.sh env:create"
    exec "${MM}" run -n winhint bash "${BASH_SOURCE[0]}" "$@"
}
heavy() { info "waiting for ${HEAVY_LOCK}"; flock "${HEAVY_LOCK}" "$@"; }

# ccache masquerade dir: compiler names → ccache (scons passes CXX as one argv).
setup_ccache() {
    local d="${WINHINT_BUILD}/ccache-bin" c
    command -v ccache >/dev/null || return 0
    mkdir -p "${d}"
    for c in "${CC:-}" "${CXX:-}"; do
        [[ -n "${c}" ]] || continue
        ln -sf "$(command -v ccache)" "${d}/$(basename "${c}")"
    done
    export PATH="${d}:${PATH}"
    export CCACHE_DIR="${CCACHE_DIR:-${WINHINT_BUILD}/ccache}"
    export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-10G}"
    export CC="$(basename "${CC}")" CXX="$(basename "${CXX}")"
}

# ── env ──────────────────────────────────────────────────────────────────────
cmd_env_create() { "${SCRIPT_DIR}/create_conda_env.sh" "$@"; }
cmd_env_update() { "${SCRIPT_DIR}/create_conda_env.sh" --update "$@"; }

# ── gem5 ─────────────────────────────────────────────────────────────────────
cmd_gem5_clone() {
    local version="" force=0 a
    for a in "$@"; do case "$a" in --force) force=1 ;; *) version="$a" ;; esac; done
    version="${version:-$(lock GEM5)}"
    local repo; repo="$(lock GEM5_REPO)"; repo="${repo:-https://github.com/gem5/gem5.git}"
    if [[ -d "${GEM5_SRC}/.git" ]]; then
        if (( ! force )); then
            ok "gem5 already cloned at ${GEM5_SRC} ($(git -C "${GEM5_SRC}" describe --tags 2>/dev/null)); --force to re-clone"
            return 0
        fi
        warn "removing ${GEM5_SRC} (builds included)"; rm -rf "${GEM5_SRC}"
    fi
    mkdir -p "${GEM5_ROOT}"
    info "cloning gem5 ${version} into ${GEM5_SRC}"
    git clone --depth 1 --branch "${version}" "${repo}" "${GEM5_SRC}"
    ok "gem5 ${version} at ${GEM5_SRC}"
}

gem5_scons() {   # $1 = variant dir name (RISCV_clean / RISCV_winhint)
    local name="$1" style=()
    # gem5's SConstruct detects GCC by "g++" in `$CXX --version`; conda's *-c++
    # alias prints "...-c++", so use the *-g++ / *-gcc names of the same compiler.
    export CXX="${CXX/%c++/g++}" CC="${CC/%-cc/-gcc}"
    setup_ccache
    # gem5 ignores CPPFLAGS/LDFLAGS (and appends *_EXTRA only after its configure
    # checks), but passes CPATH/LIBRARY_PATH through: point them at the env so
    # zlib, protobuf, hdf5… are found, and rpath the env lib dir for run time.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        export CPATH="${CONDA_PREFIX}/include${CPATH:+:${CPATH}}"
        export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
        export LINKFLAGS_EXTRA="-Wl,-rpath,${CONDA_PREFIX}/lib ${LINKFLAGS_EXTRA:-}"
    fi
    grep -q -- "--ignore-style" "${GEM5_SRC}/SConstruct" "${GEM5_SRC}"/site_scons/*.py 2>/dev/null \
        && style=(--ignore-style)
    cd "${GEM5_SRC}"
    if [[ ! -f "build/${name}/gem5.build/config" && ! -f "build/${name}/config" ]]; then
        scons defconfig "build/${name}" build_opts/RISCV "${style[@]}"
    fi
    # shellcheck disable=SC2086
    scons "build/${name}/gem5.opt" -j"${JOBS}" "${style[@]}" ${GEM5_SCONS_ARGS:-}
}

cmd_gem5_build_clean() {
    [[ -d "${GEM5_SRC}/.git" ]] || die "gem5 not cloned — run: tooling/winhint.sh gem5:clone"
    setup_ccache
    (
        info "waiting for ${HEAVY_LOCK}"; flock 9
        "${SCRIPT_DIR}/apply_patches.sh" --status | grep -q pristine \
            || die "gem5 tree is not pristine (crashed winhint build?) — run: tooling/apply_patches.sh --revert"
        gem5_scons RISCV_clean
    ) 9>"${HEAVY_LOCK}"
    ok "built ${GEM5_SRC}/build/RISCV_clean/gem5.opt"
}

cmd_gem5_build_winhint() {
    [[ -d "${GEM5_SRC}/.git" ]] || die "gem5 not cloned — run: tooling/winhint.sh gem5:clone"
    setup_ccache
    # Apply overlay + patches, build, and always restore the tree (even on failure).
    (
        info "waiting for ${HEAVY_LOCK}"; flock 9
        export WINHINT_HEAVY_LOCKED=1
        "${SCRIPT_DIR}/apply_patches.sh"
        trap '"${SCRIPT_DIR}/apply_patches.sh" --revert' EXIT
        gem5_scons RISCV_winhint
    ) 9>"${HEAVY_LOCK}"
    ok "built ${GEM5_SRC}/build/RISCV_winhint/gem5.opt"
}

cmd_gem5_build() {
    local variant="${1:-}"; shift || true
    case "${variant}" in
        clean)   cmd_gem5_build_clean "$@" ;;
        winhint) cmd_gem5_build_winhint "$@" ;;
        all)     cmd_gem5_build_clean "$@"; cmd_gem5_build_winhint "$@" ;;
        *) die "usage: tooling/winhint.sh gem5:build clean|winhint|all" ;;
    esac
}

cmd_gem5_rm() {
    case "${1:-}" in
        clean|winhint) rm -rf "${GEM5_SRC}/build/RISCV_$1"; ok "removed RISCV_$1" ;;
        all)           rm -rf "${GEM5_SRC}/build"; ok "removed all gem5 builds" ;;
        *) die "usage: tooling/winhint.sh gem5:rm clean|winhint|all" ;;
    esac
}

# ── McPAT ────────────────────────────────────────────────────────────────────
cmd_mcpat_clone() {
    local version="${1:-$(lock MCPAT)}" repo
    repo="$(lock MCPAT_REPO)"; repo="${repo:-https://github.com/HewlettPackard/mcpat.git}"
    if [[ -d "${MCPAT_SRC}/.git" ]]; then ok "McPAT already cloned at ${MCPAT_SRC}"; return 0; fi
    mkdir -p "${MCPAT_ROOT}"
    git clone --depth 1 --branch "${version}" "${repo}" "${MCPAT_SRC}"
    ok "McPAT ${version} at ${MCPAT_SRC}"
}

cmd_mcpat_build() {
    [[ -d "${MCPAT_SRC}" ]] || die "McPAT not cloned — run: tooling/winhint.sh mcpat:clone"
    setup_ccache
    # McPAT's makefile hard-codes g++; point it at the env compiler.
    heavy make -C "${MCPAT_SRC}" -j"${JOBS}" CXX="${CXX}" CC="${CC}"
    install -m 0755 "${MCPAT_SRC}/mcpat" "${MCPAT_ROOT}/mcpat"
    ok "McPAT binary at ${MCPAT_ROOT}/mcpat"
}

cmd_mcpat_rm() { rm -rf "${MCPAT_ROOT}"; ok "removed ${MCPAT_ROOT}"; }

# ── benchmarks / compiler ────────────────────────────────────────────────────
cmd_benchmarks_build() { make -C "${WINHINT_ROOT}/benchmarks" -j1 "$@"; }
cmd_benchmarks_clean() { make -C "${WINHINT_ROOT}/benchmarks" clean "$@"; }
cmd_compiler_build()   { JOBS=1 "${WINHINT_ROOT}/compiler/build.sh" "$@"; }
# B8: Clairvoyance LLVM 3.8 passes (env winhint-llvm38; takes the heavy lock itself).
cmd_clairvoyance_build() {
    [[ -f "${WINHINT_ROOT}/third_party/clairvoyance/compiler/CMakeLists.txt" || -d "${WINHINT_ROOT}/third_party/clairvoyance/compiler" ]] \
        || die "third_party/clairvoyance is empty — run: git submodule update --init third_party/clairvoyance"
    JOBS=1 "${WINHINT_ROOT}/compiler/baselines/clairvoyance/build.sh" "$@"
}

# ── real hardware (hw/, PROPOSAL Phase E2) ───────────────────────────────────
cmd_hw_build() { make -C "${WINHINT_ROOT}/hw" -j1 "$@"; }
# R2–R4 third-party baselines (opt-in; need network; no root; heavy lock inside).
cmd_hw_baselines_build() {
    local b="${1:-}"; shift || true
    local d="${WINHINT_ROOT}/hw/baselines"
    case "${b}" in
        r2) bash "${d}/r2_sched_ext/install_scx.sh" "$@" ;;
        r3) bash "${d}/r3_lpmd/install_lpmd.sh" "$@" ;;
        r4) bash "${d}/r4_pie/install_pmctrack.sh" "${@:-build}" ;;
        *)  die "usage: tooling/winhint.sh hw-baselines:build r2|r3|r4 [args]" ;;
    esac
}

# ── documentation (env winhint, requirements-docs.txt, mkdocs.yml) ──
# The site (guides + Python API via mkdocstrings + C/C++ API via mkdoxy/Doxygen) is
# built with --strict: a broken link, a bad docstring or a missing page fails the build.
docs_run() {
    [[ -d "${MAMBA_ROOT_PREFIX}/envs/winhint/conda-meta" ]] \
        || die "micromamba env 'winhint' not found — run: tooling/create_conda_env.sh"
    (cd "${WINHINT_ROOT}" && DISABLE_MKDOCS_2_WARNING=true "${MM}" run -n winhint "$@")
}
cmd_docs_build() {
    docs_run mkdocs build --strict --site-dir "${WINHINT_BUILD}/site" "$@"
    ok "site: ${WINHINT_BUILD}/site/index.html"
}
cmd_docs_serve() { docs_run mkdocs serve --strict "$@"; }
cmd_docs_check() { docs_run python tooling/check_docstrings.py --require-griffe "$@"; }

# ── tests ────────────────────────────────────────────────────────────────────
PYTEST_DIRS=(sim/baselines/lut/tests sim/baselines/oracle/tests sim/baselines/tune/tests sim/fidelity/tests sim/tests analysis/tests hw/tests tooling/tests compiler/baselines/pgo/tests)
cmd_test() {
    local what="${1:-all}"; shift || true
    case "${what}" in
        env)      "${SCRIPT_DIR}/create_conda_env.sh" --verify-only ;;
        python)   # One pytest process per directory: several tests/ dirs have their own
                  # conftest.py and import helpers from it (`from conftest import ...`),
                  # which collides when they are collected in a single session.
                  local t rc=0 failed=()
                  for t in "${PYTEST_DIRS[@]}"; do
                      [[ -d "${WINHINT_ROOT}/${t}" ]] || continue
                      info "pytest ${t}"
                      (cd "${WINHINT_ROOT}" && python -m pytest -q -p no:cacheprovider "${t}" "$@") \
                          || { rc=1; failed+=("${t}"); }
                  done
                  (( rc == 0 )) || die "pytest failed in: ${failed[*]}"
                  ok "pytest: all ${#PYTEST_DIRS[@]} test dirs passed" ;;
        compiler) "${WINHINT_ROOT}/compiler/test/run_tests.sh" "$@" ;;
        sim)      "${WINHINT_ROOT}/sim/tests/run_tests.sh" "$@" ;;   # needs both gem5 builds; heavy lock
        docs)     cmd_docs_check && cmd_docs_build ;;
        all)      cmd_test env; cmd_test python; cmd_test compiler; cmd_test docs ;;
        *)        die "usage: tooling/winhint.sh test env|python|compiler|sim|docs|all" ;;
    esac
}

# ── coverage gate and commit gate ────────────────────────────────────────────
# tooling/coverage_gate.py: Python, C++ and C line coverage on instrumented host builds; each
# language must be strictly above 90 % (out-of-scope code: .coveragerc, CXX_EXCLUDED).
cmd_coverage() { python "${SCRIPT_DIR}/coverage_gate.py" "$@"; }
# The commit gate (make validate, the pre-commit hook): toolchain check, every host test
# suite under the coverage gate, docstring check and strict docs build. No gem5 needed.
cmd_validate() {
    info "validate 1/3: environment";       cmd_test env
    info "validate 2/3: tests + coverage";  cmd_coverage
    info "validate 3/3: documentation";     cmd_test docs
    ok "validate: all gates passed"
}

# ── verification (PROPOSAL §7) ───────────────────────────────────────────────
# Every hinted variant in build/benchmarks vs plain: qemu-riscv64, native x86 and
# gem5 RISCV_clean (when built; heavy lock) → results/correctness.csv.
cmd_verify() { python "${SCRIPT_DIR}/verify_correctness.py" "$@"; }

# ── status ───────────────────────────────────────────────────────────────────
row() { printf '  %-22s %b\n' "$1" "$2"; }
have() { local v; if v="$("$@" 2>/dev/null | head -1)" && [[ -n "${v}" ]]; then echo -e "${GREEN}${v}${NC}"; else echo -e "${RED}missing${NC}"; fi; }
file_row() { if [[ -e "$2" ]]; then row "$1" "${GREEN}$2${NC}"; else row "$1" "${YELLOW}not built${NC} ($2)"; fi; }

cmd_status() {
    info "WinHint status (WINHINT_ROOT=${WINHINT_ROOT}, WINHINT_BUILD=${WINHINT_BUILD})"
    row "env winhint"        "${GREEN}${CONDA_PREFIX}${NC}"
    row "python"             "$(have python -c 'import sys,torch,pandas; print(sys.version.split()[0], "torch", torch.__version__, "pandas", pandas.__version__)')"
    row "clang"              "$(have bash -c 'clang --version | head -1')"
    row "llvm-config"        "$(have llvm-config --version)"
    row "lit / FileCheck"    "$(have bash -c 'lit --version 2>&1 | head -1; command -v FileCheck >/dev/null')"
    row "host g++"           "$(have bash -c '"$CXX" --version')"
    row "scons"              "$(have bash -c 'scons --version | grep -m1 "SCons:" | grep -o "v[0-9][0-9.]*[0-9]" | head -1')"
    row "riscv gcc"          "$(have riscv64-conda-linux-gnu-gcc -dumpfullversion)"
    row "riscv clang"        "$(have bash -c 'echo "int x;" | clang ${WINHINT_RISCV_CLANG_FLAGS:-} -x c -c - -o /dev/null && echo "clang --target=${WINHINT_RISCV_TRIPLE:-?} ok"')"
    row "qemu-riscv64"       "$(have qemu-riscv64 --version)"
    row "make/cmake/ninja"   "$(have bash -c 'echo "make $(make --version | head -1 | grep -o "[0-9][0-9.]*$"), cmake $(cmake --version | head -1 | grep -o "[0-9][0-9.]*"), ninja $(ninja --version)"')"
    row "m4 / pkg-config"    "$(have bash -c 'echo "m4 $(m4 --version | head -1 | grep -o "[0-9][0-9.]*$"), pkg-config $(pkg-config --version)"')"
    row "ccache / git"       "$(have bash -c 'echo "ccache $(ccache --version | head -1 | grep -o "[0-9][0-9.]*$"), git $(git --version | grep -o "[0-9][0-9.]*")"')"
    row "rustc / cargo"      "$(have bash -c 'echo "$(rustc --version | cut -d" " -f2) / $(cargo --version | cut -d" " -f2)"')"
    row "perf"               "$(have perf --version)"
    row "bpftool"            "$(have bash -c 'b=$(command -v bpftool) && echo "$(bpftool version | head -1) ($b; not from conda)"')"
    if [[ -d "${MAMBA_ROOT_PREFIX}/envs/winhint-llvm38/conda-meta" ]]; then
        row "env winhint-llvm38" "$(have "${MM}" run -n winhint-llvm38 llvm-config --version)"
    else
        row "env winhint-llvm38" "${YELLOW}absent${NC} (B8 → port)"
    fi
    if [[ -d "${MAMBA_ROOT_PREFIX}/envs/winhint-kmod/conda-meta" ]]; then
        row "env winhint-kmod" "$(have "${MAMBA_ROOT_PREFIX}/envs/winhint-kmod/bin/x86_64-conda-linux-gnu-gcc" -dumpfullversion)"
    else
        row "env winhint-kmod" "${YELLOW}absent${NC}"
    fi
    if [[ -d "${GEM5_SRC}/.git" ]]; then
        local g5; g5="$(git -C "${GEM5_SRC}" describe --tags 2>/dev/null || true)"
        if [[ "${g5}" == "$(lock GEM5)"* ]]; then
            row "gem5 source" "${GREEN}${GEM5_SRC} (${g5})${NC}"
        else
            row "gem5 source" "${RED}${GEM5_SRC} (${g5:-unknown} != versions.lock GEM5=$(lock GEM5))${NC}"
        fi
        row "gem5 tree"   "$("${SCRIPT_DIR}/apply_patches.sh" --status 2>&1 | head -1 | sed 's/^ *//')"
    else
        row "gem5 source" "${YELLOW}not cloned${NC}"
    fi
    file_row "gem5 RISCV_clean"   "${GEM5_SRC}/build/RISCV_clean/gem5.opt"
    file_row "gem5 RISCV_winhint" "${GEM5_SRC}/build/RISCV_winhint/gem5.opt"
    if [[ -d "${MCPAT_SRC}/.git" ]]; then
        row "McPAT source" "${GREEN}${MCPAT_SRC} ($(git -C "${MCPAT_SRC}" describe --tags --always 2>/dev/null))${NC}"
    else
        row "McPAT source" "${YELLOW}not cloned${NC}"
    fi
    file_row "McPAT"              "${MCPAT_ROOT}/mcpat"
    file_row "WinHint.so"         "${WINHINT_BUILD}/compiler/WinHint.so"
    file_row "JonesIQ.so"         "${WINHINT_BUILD}/compiler/JonesIQ.so"
    if [[ -n "$(ls -A "${WINHINT_ROOT}/third_party/clairvoyance" 2>/dev/null)" ]]; then
        row "clairvoyance src" "${GREEN}third_party/clairvoyance ($(git -C "${WINHINT_ROOT}/third_party/clairvoyance" rev-parse --short HEAD 2>/dev/null))${NC}"
    else
        row "clairvoyance src" "${YELLOW}submodule not initialized${NC}"
    fi
    local n
    n="$(ls "${WINHINT_BUILD}"/clairvoyance/lib/*.so 2>/dev/null | wc -l)"
    if (( n > 0 )); then row "clairvoyance passes" "${GREEN}${n} .so in ${WINHINT_BUILD}/clairvoyance/lib${NC}"
    else row "clairvoyance passes" "${YELLOW}not built${NC} (${WINHINT_BUILD}/clairvoyance/lib)"; fi
    n="$(find "${WINHINT_BUILD}/benchmarks" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)"
    if (( n > 0 )); then
        row "benchmarks" "${GREEN}${WINHINT_BUILD}/benchmarks (${n} arch/variant dirs: $(cd "${WINHINT_BUILD}/benchmarks" && ls -d */* 2>/dev/null | head -8 | tr '\n' ' '))${NC}"
    else
        row "benchmarks" "${YELLOW}not built${NC} (${WINHINT_BUILD}/benchmarks)"
    fi
    file_row "libwinhint"         "${WINHINT_BUILD}/libwinhint/libwinhint.so"
    file_row "hw tools"           "${WINHINT_BUILD}/hw/wh_measure"
    file_row "R2 sched_ext"       "${WINHINT_BUILD}/hw-baselines/scx/target/release/scx_bpfland"
    file_row "R3 intel-lpmd"      "${WINHINT_BUILD}/hw-baselines/lpmd/prefix/sbin/intel_lpmd"
    file_row "R4 PMCTrack CLI"    "${WINHINT_BUILD}/hw-baselines/pmctrack/bin/pmctrack"
    if [[ -d "${WINHINT_ROOT}/results" ]]; then
        row "results" "${GREEN}${WINHINT_ROOT}/results ($(ls "${WINHINT_ROOT}/results" | tr '\n' ' '))${NC}"
    else
        row "results" "${YELLOW}empty${NC}"
    fi
    if [[ -e "${HEAVY_LOCK}" ]] && ! flock -n "${HEAVY_LOCK}" true; then
        row "heavy lock" "${YELLOW}held (a heavy job is running)${NC}"
    else
        row "heavy lock" "free"
    fi
}

cmd_help() {
    cat <<EOF
tooling/winhint.sh — WinHint project manager (host + micromamba, no Docker)

Usage: tooling/winhint.sh <command> [args]

Environment
  env:create [opts]          Create the conda env(s) (tooling/create_conda_env.sh [opts])
  env:update [opts]          Sync env winhint with environment.yml and requirements*.txt

gem5 (source: build/gem5/src, version from tooling/versions.lock)
  gem5:clone [tag] [--force] Shallow-clone gem5 (no-op if present; --force re-clones)
  gem5:build clean           Build build/gem5/src/build/RISCV_clean/gem5.opt (unmodified)
  gem5:build winhint         Copy the sim/gem5/ overlay, apply sim/patches/gem5_*_*.patch
                             (name order, tooling/apply_patches.sh), build
                             build/gem5/src/build/RISCV_winhint/gem5.opt, restore the tree
  gem5:build all             clean, then winhint
  gem5:rm clean|winhint|all  Remove gem5 build(s)

McPAT (build/mcpat)
  mcpat:clone [tag]          Clone McPAT
  mcpat:build                Build build/mcpat/mcpat
  mcpat:rm                   Remove McPAT

Workloads and compiler
  benchmarks:build [make args]  make -C benchmarks (e.g. ARCH=riscv VARIANT=winhint)
  benchmarks:clean
  compiler:build [args]      compiler/build.sh → build/compiler/{WinHint,JonesIQ}.so
  clairvoyance:build         B8: Clairvoyance passes (env winhint-llvm38) → build/clairvoyance/lib

Real hardware (hw/, see docs/guide/hardware/index.md)
  hw:build [make args]       make -C hw → build/libwinhint/, build/hw/
  hw-baselines:build r2|r3|r4 [args]
                             Opt-in builds of sched_ext (R2), intel-lpmd (R3), PMCTrack (R4;
                             default arg: build) → build/hw-baselines/ (network, no root)

Documentation (env winhint, requirements-docs.txt; site source: mkdocs.yml, docs/)
  docs:build [args]          Strict MkDocs build → build/site (guides, Python + C/C++ API)
  docs:serve [args]          Live-reloading preview at http://127.0.0.1:8000
  docs:check [files]         Docstring coverage + Google-style syntax (tooling/check_docstrings.py)

Commit gate
  validate                   env check + every host test suite under the coverage gate + docs
                             (= make validate; the pre-commit hook runs it: make install-hooks)
  coverage [python|cxx|c] [--no-gate]
                             Line coverage on instrumented host builds; each language must be
                             > 90 % (tooling/coverage_gate.py) → build/coverage/

Verification
  verify [args]              Bit-identical output of every hinted variant vs plain under
                             qemu-riscv64, native x86 and gem5 RISCV_clean (if built; heavy
                             lock, small inputs) → results/correctness.csv
                             (tooling/verify_correctness.py --help)
  test env|python|compiler|sim|docs|all
                             env: create_conda_env.sh --verify-only; python: pytest on every tests/ dir;
                             compiler: compiler/test/run_tests.sh; sim: sim/tests/run_tests.sh
                             (needs both gem5 builds; heavy lock); docs: docs:check + docs:build;
                             all = env + python + compiler + docs

  status                     Report every tool and build
  help                       This help

Heavy jobs (gem5/McPAT builds) run under flock \$WINHINT_BUILD/.heavy.lock with
-j\${JOBS:-2} (capped at 2). Extra scons args: GEM5_SCONS_ARGS="...".
ccache cache: \$WINHINT_BUILD/ccache.
EOF
}

main() {
    local cmd="${1:-help}"; shift || true
    case "${cmd}" in
        help|-h|--help) cmd_help; return 0 ;;
        env:create)     cmd_env_create "$@"; return ;;
        env:update)     cmd_env_update "$@"; return ;;
        docs:build)     cmd_docs_build "$@"; return ;;
        docs:serve)     cmd_docs_serve "$@"; return ;;
        docs:check)     cmd_docs_check "$@"; return ;;
    esac
    ensure_env "${cmd}" "$@"
    mkdir -p "${WINHINT_BUILD}"
    case "${cmd}" in
        gem5:clone)       cmd_gem5_clone "$@" ;;
        gem5:build)       cmd_gem5_build "$@" ;;
        gem5:rm)          cmd_gem5_rm "$@" ;;
        mcpat:clone)      cmd_mcpat_clone "$@" ;;
        mcpat:build)      cmd_mcpat_build "$@" ;;
        mcpat:rm)         cmd_mcpat_rm "$@" ;;
        benchmarks:build) cmd_benchmarks_build "$@" ;;
        benchmarks:clean) cmd_benchmarks_clean "$@" ;;
        compiler:build)   cmd_compiler_build "$@" ;;
        clairvoyance:build) cmd_clairvoyance_build "$@" ;;
        hw:build)         cmd_hw_build "$@" ;;
        hw-baselines:build) cmd_hw_baselines_build "$@" ;;
        test)             cmd_test "$@" ;;
        verify|verify:correctness) cmd_verify "$@" ;;
        coverage)         cmd_coverage "$@" ;;
        validate)         cmd_validate "$@" ;;
        status)           cmd_status "$@" ;;
        *) die "unknown command: ${cmd} (tooling/winhint.sh help)" ;;
    esac
}

main "$@"
