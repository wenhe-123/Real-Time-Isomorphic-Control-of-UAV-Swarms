"""Compatibility shim: re-exports morph geometry from the active sampling backend.

See :func:`functions.open_close.sampling_backend.get_sampling_backend_name` for
backend selection via ``ISO_SWARM_SAMPLING_BACKEND``.
"""

from __future__ import annotations

from functions.open_close.sampling_backend import load_sampling_backend_module

_impl = load_sampling_backend_module("morph_geometry.py", module_key="sampling_backend_morph_geometry")

for _name, _value in vars(_impl).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
