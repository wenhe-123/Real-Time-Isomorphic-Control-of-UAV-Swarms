#!/usr/bin/env bash
# Pixi deploy: stdlib _sqlite3 needs libsqlite with sqlite3_deserialize (IPython/swarm_gpt.core).
_root="${CONDA_PREFIX:-${PIXI_ENVIRONMENT:-}}"
if [ -z "$_root" ]; then
  return 0 2>/dev/null || exit 0
fi
_lib="$(ls -1 "${_root}/lib"/libsqlite3.so.3.* 2>/dev/null | sort -V | tail -1)"
if [ -n "$_lib" ] && [ -f "$_lib" ]; then
  case ":${LD_PRELOAD:-}:" in
    *":$_lib:"*) ;;
    *) export LD_PRELOAD="$_lib${LD_PRELOAD:+:$LD_PRELOAD}" ;;
  esac
fi
