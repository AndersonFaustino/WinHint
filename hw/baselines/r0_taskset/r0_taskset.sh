#!/usr/bin/env bash
# R0: run a command pinned to all P-cores or all E-cores.
#   r0_taskset.sh P|E -- cmd [args...]
. "$(dirname "$0")/../common.sh"
side="${1:-}"; shift || true
[ "${1:-}" = "--" ] && shift
wh_require_hybrid
case "$side" in
  P|p) cpus="$(wh_pcpus)" ;;
  E|e) cpus="$(wh_ecpus)" ;;
  *) wh_die "usage: $0 P|E -- cmd [args...]" ;;
esac
[ -n "$cpus" ] || wh_die "could not determine the $side-core list (set WINHINT_PCPUS/WINHINT_ECPUS)"
exec taskset -c "$cpus" "$@"
