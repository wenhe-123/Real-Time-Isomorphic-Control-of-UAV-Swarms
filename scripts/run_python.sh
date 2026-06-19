#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${PIXI_ENVIRONMENT_NAME:-default}"
CONDA_PREFIX="${ROOT}/.pixi/envs/${ENV}"
if [ "$ENV" = "deploy" ] && [ -f "${ROOT}/scripts/ensure_sqlite.sh" ]; then
  # shellcheck source=scripts/ensure_sqlite.sh
  source "${ROOT}/scripts/ensure_sqlite.sh"
fi
exec "${CONDA_PREFIX}/bin/python" "$@"
