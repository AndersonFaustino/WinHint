#!/usr/bin/env bash
# =============================================================================
# tooling/apply_patches.sh — put the WinHint changes into the gem5 tree, or
# take them out again.
#
# "apply" does two things, in this order:
#   1. copies the overlay sim/gem5/ (new source files, e.g.
#      sim/gem5/src/cpu/o3/window/*) into the gem5 tree (same relative paths);
#   2. applies every sim/patches/gem5_*_*.patch in name order (git apply).
# "--revert" restores the gem5 tree to its pristine git state (tracked files
# checked out, untracked files removed; build/ is never touched).
#
# Usage:
#   tooling/apply_patches.sh            apply overlay + patches
#   tooling/apply_patches.sh --check    dry-run: do the patches apply cleanly?
#   tooling/apply_patches.sh --revert   restore the pristine gem5 tree
#   tooling/apply_patches.sh --status   show whether the tree is modified
#   GEM5_DIR=/path/to/gem5 tooling/apply_patches.sh ...
#
# Default GEM5_DIR: $WINHINT_BUILD/gem5/src (WINHINT_BUILD defaults to
# <repo>/build). `tooling/winhint.sh gem5:build winhint` calls this script
# automatically (apply, build, revert).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WINHINT_ROOT="${WINHINT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WINHINT_BUILD="${WINHINT_BUILD:-${WINHINT_ROOT}/build}"
GEM5_DIR="${GEM5_DIR:-${WINHINT_BUILD}/gem5/src}"
OVERLAY_DIR="${OVERLAY_DIR:-${WINHINT_ROOT}/sim/gem5}"
PATCH_DIR="${PATCH_DIR:-${WINHINT_ROOT}/sim/patches}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }

MODE="apply"
case "${1:-}" in
    "")        ;;
    --check)   MODE="check" ;;
    --revert)  MODE="revert" ;;
    --status)  MODE="status" ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
esac

# Modifying the shared gem5 tree must not race a gem5 build: take the heavy
# lock unless the caller (tooling/winhint.sh gem5:build) already holds it.
if [[ "${MODE}" == "apply" || "${MODE}" == "revert" ]] && [[ "${WINHINT_HEAVY_LOCKED:-0}" != 1 ]]; then
    mkdir -p "${WINHINT_BUILD}"
    export WINHINT_HEAVY_LOCKED=1
    exec flock "${WINHINT_BUILD}/.heavy.lock" "$0" "$@"
fi

[[ -d "${GEM5_DIR}/.git" ]] || die "gem5 git tree not found at ${GEM5_DIR} (run: tooling/winhint.sh gem5:clone)"

shopt -s nullglob
PATCHES=("${PATCH_DIR}"/gem5_*_*.patch)   # glob expansion is sorted by name
shopt -u nullglob

g() { git -C "${GEM5_DIR}" "$@"; }
tree_dirty() { [[ -n "$(g status --porcelain --untracked-files=all -- . ':!build' ':!m5out')" ]]; }

case "${MODE}" in
    status)
        if tree_dirty; then
            warn "gem5 tree at ${GEM5_DIR} is modified:"; g status --short -- . ':!build' | head -40
        else
            ok "gem5 tree at ${GEM5_DIR} is pristine"
        fi ;;
    check)
        (( ${#PATCHES[@]} )) || { warn "no sim/patches/gem5_*_*.patch found"; exit 0; }
        tree_dirty && warn "tree is not pristine; checking against HEAD"
        # Patches stack (zz_ltp needs the winhint patch), so apply them in order
        # to a throwaway index; the working tree and the real index are untouched.
        tmp_index="$(mktemp)"; trap 'rm -f "${tmp_index}"' EXIT
        GIT_INDEX_FILE="${tmp_index}" g read-tree HEAD
        for p in "${PATCHES[@]}"; do
            GIT_INDEX_FILE="${tmp_index}" g apply --cached "${p}" \
                || die "$(basename "${p}") does not apply to ${GEM5_DIR} (after the previous patches)"
            ok "$(basename "${p}") applies cleanly"
        done ;;
    apply)
        tree_dirty && die "gem5 tree is not pristine (run: tooling/apply_patches.sh --revert)"
        if [[ -d "${OVERLAY_DIR}" ]]; then
            n="$(cd "${OVERLAY_DIR}" && find . -type f ! -name '*.orig' ! -name '*.rej' | wc -l)"
            (cd "${OVERLAY_DIR}" && find . -type f ! -name '*.orig' ! -name '*.rej' -print0 \
                | xargs -0 -r cp --parents -t "${GEM5_DIR}")
            ok "copied ${n} overlay file(s) from ${OVERLAY_DIR#${WINHINT_ROOT}/}"
        fi
        for p in "${PATCHES[@]}"; do
            if ! g apply --whitespace=nowarn "${p}"; then
                g checkout -q -- . ; g clean -fdq -e build -e m5out
                die "$(basename "${p}") failed to apply; gem5 tree restored"
            fi
            ok "applied $(basename "${p}")"
        done
        (( ${#PATCHES[@]} )) || warn "no sim/patches/gem5_*_*.patch found (overlay only)" ;;
    revert)
        g checkout -q -- .
        g clean -fdq -e build -e m5out
        ok "gem5 tree at ${GEM5_DIR} restored to $(g describe --tags --always 2>/dev/null)" ;;
esac
