"""Compatibility shim for sampling backend morph LP plotting."""

from __future__ import annotations

from shared.sampling_backend import load_sampling_backend_module

_impl = load_sampling_backend_module("morph_lp_plot.py", module_key="sampling_backend_morph_lp_plot")

for _name, _value in vars(_impl).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
