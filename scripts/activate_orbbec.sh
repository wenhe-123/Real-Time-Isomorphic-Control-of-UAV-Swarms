#!/usr/bin/env bash
# Pixi activation: wire ORBBEC_ROOT into PATH / LD_LIBRARY_PATH when SDK is present.
if [[ -z "${ORBBEC_ROOT:-}" && -n "${PIXI_PROJECT_ROOT:-}" ]]; then
  _local="${PIXI_PROJECT_ROOT}/third_party/orbbec/current"
  if [[ -d "${_local}/lib" ]]; then
    export ORBBEC_ROOT="${_local}"
  fi
fi

if [[ -n "${ORBBEC_ROOT:-}" && -d "${ORBBEC_ROOT}/lib" ]]; then
  export PATH="${ORBBEC_ROOT}/bin:${PATH}"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${ORBBEC_ROOT}/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${ORBBEC_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi
