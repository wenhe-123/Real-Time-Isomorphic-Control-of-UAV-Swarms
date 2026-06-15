"""Compatibility shim for sampling backend morph geometry."""

from __future__ import annotations

from functions.open_close.sampling_backend import load_sampling_backend_module

_impl = load_sampling_backend_module("morph_geometry.py", module_key="sampling_backend_morph_geometry")

for _name, _value in vars(_impl).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
