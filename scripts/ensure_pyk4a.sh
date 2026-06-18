#!/usr/bin/env bash
# Pixi activation: install pyk4a wheel when Orbbec SDK is present (pip, not source build).
if python -c "import pyk4a" 2>/dev/null; then
  exit 0
fi
ROOT="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -d "${ROOT}/third_party/orbbec/current/include" ]]; then
  bash "${ROOT}/scripts/install_pyk4a.sh"
fi
