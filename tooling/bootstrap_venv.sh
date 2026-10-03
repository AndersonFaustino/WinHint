#!/usr/bin/env bash
# Create (if needed) and update the conda environment that `make validate` runs in.
#
#   tooling/bootstrap_venv.sh
#
# The commit gate (`make validate`, run by the pre-commit hook) depends on this, so a fresh
# clone validates its first commit without any manual setup beyond curl and the network:
# tooling/create_conda_env.sh downloads micromamba when no conda front end is installed.
#
# Idempotent: the `winhint` environment is created when it is missing, and updated
# (create_conda_env.sh --update) only when environment.yml, requirements*.txt or
# tooling/versions.lock changed since the last successful run (their hash is kept in
# $WINHINT_BUILD/.env-installed). An existing environment with no stamp is only stamped.
# The gate does not need winhint-llvm38 or winhint-kmod, so they are not created here
# (tooling/create_conda_env.sh creates them).
#
# Environment: MAMBA_ROOT_PREFIX (default ~/.local/share/mamba), WINHINT_BUILD (default <repo>/build).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD="${WINHINT_BUILD:-${ROOT}/build}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.local/share/mamba}"
STAMP="${BUILD}/.env-installed"
OPTS=(--extras dev,docs --skip-llvm38 --skip-kmod)

say() { echo "==> [env] $*"; }

wanted="$(cat "${ROOT}/environment.yml" "${ROOT}"/requirements*.txt "${SCRIPT_DIR}/versions.lock" \
              | sha256sum | cut -d' ' -f1)"

mkdir -p "${BUILD}"
if [[ ! -d "${MAMBA_ROOT_PREFIX}/envs/winhint/conda-meta" ]]; then
    say "creating the winhint environment (first run; this takes a while)"
    "${SCRIPT_DIR}/create_conda_env.sh" "${OPTS[@]}"
elif [[ -f "${STAMP}" && "$(cat "${STAMP}")" != "${wanted}" ]]; then
    say "environment files changed: updating the winhint environment"
    "${SCRIPT_DIR}/create_conda_env.sh" --update "${OPTS[@]}"
elif [[ -f "${STAMP}" ]]; then
    exit 0
fi
echo "${wanted}" > "${STAMP}"
say "ready"
