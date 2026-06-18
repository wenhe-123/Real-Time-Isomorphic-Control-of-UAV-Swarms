#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${PIXI_ENVIRONMENT_NAME:-default}"
exec "${ROOT}/.pixi/envs/${ENV}/bin/python" "$@"
