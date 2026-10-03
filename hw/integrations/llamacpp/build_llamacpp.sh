#!/usr/bin/env bash
# hw/integrations/llamacpp/build_llamacpp.sh -- llama.cpp with WinHint operator/layer hooks
# (PROPOSAL Phase E2, optional). Host only, conda env `winhint` (cmake, ninja, conda GCC).
#
#   bash hw/integrations/llamacpp/build_llamacpp.sh [fetch|build|model|check|all]   (default: all)
#
#   fetch  clone llama.cpp at the pinned tag into build/integrations/llamacpp/src, apply the patch
#   build  two static builds of llama-simple (under the heavy lock, -j2):
#            build/integrations/llamacpp/build-vanilla  (unpatched code paths, GGML_WINHINT=OFF)
#            build/integrations/llamacpp/build-winhint  (GGML_WINHINT=ON, linked with libwinhint.a)
#   model  download the tiny test model stories15M-q4_0.gguf (19 MB, SHA-256 checked)
#   check  functional check (no measurement): identical greedy tokens for vanilla, and winhint with
#          WINHINT_MODE=off / log, GGML_WINHINT_REGIONS=op / layer; libwinhint's per-region
#          log must list the operator / layer regions.
#
# Env: LLAMACPP_TAG (default below), WINHINT_BUILD, JOBS (default 2).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WINHINT_ROOT="${WINHINT_ROOT:-$(cd "$HERE/../../.." && pwd)}"
WINHINT_BUILD="${WINHINT_BUILD:-$WINHINT_ROOT/build}"
LLAMACPP_TAG="${LLAMACPP_TAG:-b11327}"           # pinned; the patch is tested against this tag
LLAMACPP_URL="${LLAMACPP_URL:-https://github.com/ggml-org/llama.cpp}"
PATCH="$HERE/winhint-ggml-hooks.patch"
OUT="$WINHINT_BUILD/integrations/llamacpp"
SRC="$OUT/src"
JOBS="${JOBS:-2}"
LIBWH_DIR="${LIBWINHINT_DIR:-$WINHINT_BUILD/libwinhint}"
MODEL_URL="https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf"
MODEL_SHA="66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739"
MODEL="$OUT/models/stories15M-q4_0.gguf"

log() { printf '[build_llamacpp] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
[ "$(id -u)" -ne 0 ] || die "build as your normal user, not root"
command -v cmake >/dev/null && command -v ninja >/dev/null || \
  die "cmake/ninja not found: activate the winhint env (build/ENV.md)"

fetch() {
  mkdir -p "$OUT"
  if [ ! -d "$SRC/.git" ]; then
    git clone -q --depth 1 --branch "$LLAMACPP_TAG" "$LLAMACPP_URL" "$SRC"
  fi
  local have; have="$(git -C "$SRC" describe --tags --exact-match 2>/dev/null || true)"
  [ "$have" = "$LLAMACPP_TAG" ] || die "$SRC is at '$have', expected $LLAMACPP_TAG (remove it to re-clone)"
  if git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
    log "patch already applied"
  else
    git -C "$SRC" apply "$PATCH"
    log "patch applied"
  fi
  git -C "$SRC" rev-parse HEAD > "$OUT/WINHINT_LLAMACPP_COMMIT"
}

build_one() {   # <name> <extra cmake args...>
  local name="$1"; shift
  cmake -S "$SRC" -B "$OUT/build-$name" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=ON "$@" >/dev/null
  flock "$WINHINT_BUILD/.heavy.lock" cmake --build "$OUT/build-$name" -j "$JOBS" --target llama-simple
  log "built $OUT/build-$name/bin/llama-simple"
}

build() {
  [ -f "$SRC/ggml/src/ggml-cpu/winhint-hooks.h" ] || fetch
  [ -f "$LIBWH_DIR/libwinhint.a" ] || die "$LIBWH_DIR/libwinhint.a missing: make -C hw -j1"
  build_one vanilla -DGGML_WINHINT=OFF
  # libwinhint uses perf/RAPL helpers only (libc); -lm for safety.
  build_one winhint -DGGML_WINHINT=ON -DGGML_WINHINT_INCLUDE="$WINHINT_ROOT/hw/libwinhint" \
    -DGGML_WINHINT_LIB="$LIBWH_DIR/libwinhint.a"
}

model() {
  mkdir -p "$(dirname "$MODEL")"
  if [ ! -f "$MODEL" ]; then
    curl -fsSL -o "$MODEL.part" "$MODEL_URL"
    mv "$MODEL.part" "$MODEL"
  fi
  echo "$MODEL_SHA  $MODEL" | sha256sum -c --quiet - || die "checksum mismatch for $MODEL"
}

# Greedy decode, 1 thread (libwinhint migrates the calling thread only), stdout = tokens.
run_simple() {   # <build> -> generated text on stdout
  "$OUT/build-$1/bin/llama-simple" -m "$MODEL" -n "${N_PREDICT:-48}" "${PROMPT:-Once upon a time}" \
    2>/dev/null
}

check() {
  [ -f "$MODEL" ] || model
  local d="$OUT/check"; mkdir -p "$d"; rm -f "$d"/*
  local fail=0
  run_simple vanilla > "$d/vanilla.txt"
  [ -s "$d/vanilla.txt" ] || die "vanilla produced no output"
  WINHINT_MODE=off                                      run_simple winhint > "$d/off.txt"
  WINHINT_MODE=log WINHINT_LOG="$d/op.csv"              run_simple winhint > "$d/log_op.txt"
  WINHINT_MODE=log WINHINT_LOG="$d/layer.csv" GGML_WINHINT_REGIONS=layer \
                                                        run_simple winhint > "$d/log_layer.txt"
  WINHINT_MODE=log WINHINT_LOG="$d/regions.csv" GGML_WINHINT_SETWIN=none \
                                                        run_simple winhint > "$d/log_regions.txt"
  for f in off log_op log_layer log_regions; do
    if cmp -s "$d/vanilla.txt" "$d/$f.txt"; then echo "[SAME] $f"; else echo "[DIFF] $f"; fail=1; fi
  done
  # op mode: regions 1 (matmul) and 2 (norm) must appear; layer mode: region 16 (layer 0).
  for spec in "op.csv:1" "op.csv:2" "layer.csv:16" "regions.csv:1"; do
    local f="${spec%%:*}" r="${spec#*:}"
    if [ -f "$d/$f" ] && awk -F, -v r="$r" 'NR>1 && $1==r && $3>0 {ok=1} END {exit !ok}' "$d/$f"; then
      echo "[OK]   $f lists region $r"
    else
      echo "[FAIL] $f has no region $r"; fail=1
    fi
  done
  log "outputs and libwinhint logs in $d"
  return $fail
}

case "${1:-all}" in
  fetch) fetch ;;
  build) build ;;
  model) model ;;
  check) check ;;
  all)   fetch; build; model; check ;;
  *) die "usage: $0 [fetch|build|model|check|all]" ;;
esac
