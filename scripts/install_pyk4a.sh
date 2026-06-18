#!/usr/bin/env bash
# Build pyk4a against Orbbec K4A Wrapper headers (requires ORBBEC_ROOT or third_party/orbbec/current).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${ORBBEC_ROOT:-}" ]]; then
  _local="${ROOT}/third_party/orbbec/current"
  if [[ -d "${_local}/include" || -d "${_local}/lib" ]]; then
    export ORBBEC_ROOT="${_local}"
  fi
fi

if [[ -z "${ORBBEC_ROOT:-}" || ! -d "${ORBBEC_ROOT}" ]]; then
  echo "ORBBEC_ROOT not set and ${ROOT}/third_party/orbbec/current missing." >&2
  echo "Run: pixi run setup-orbbec   (or export ORBBEC_ROOT=...)" >&2
  exit 1
fi

export CMAKE_PREFIX_PATH="${ORBBEC_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export CPATH="${ORBBEC_ROOT}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${ORBBEC_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CFLAGS="-I${ORBBEC_ROOT}/include ${CFLAGS:-}"
export LDFLAGS="-L${ORBBEC_ROOT}/lib ${LDFLAGS:-}"

_ENV_NAME="${1:-${PIXI_ENVIRONMENT_NAME:-default}}"
PYTHON="${PIXI_ENVIRONMENT:-${ROOT}/.pixi/envs/${_ENV_NAME}}"
PYTHON="${PYTHON%/}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "Pixi python not found at ${ROOT}/.pixi/envs/${_ENV_NAME}/bin/python" >&2
  echo "Run: pixi install   then: pixi run setup" >&2
  exit 1
fi

echo "Installing pyk4a into $("${PYTHON}" -c 'import sys; print(sys.prefix)') (ORBBEC_ROOT=${ORBBEC_ROOT}) ..."
"${PYTHON}" -m pip install "pyk4a>=1.5.0"
