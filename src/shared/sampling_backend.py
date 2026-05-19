"""Sampling backend loader for point-mapping modules.

Backends:
- ``pure_angular`` (default)
- ``3region_mapping``

Set env ``ISO_SWARM_SAMPLING_BACKEND`` to switch.
"""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

_VALID_BACKENDS = {"pure_angular", "3region_mapping"}


def get_sampling_backend_name() -> str:
    name = str(os.getenv("ISO_SWARM_SAMPLING_BACKEND", "pure_angular")).strip()
    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid ISO_SWARM_SAMPLING_BACKEND='{name}'. "
            f"Expected one of: {sorted(_VALID_BACKENDS)}"
        )
    return name


def load_sampling_backend_module(module_filename: str, *, module_key: str) -> ModuleType:
    here = Path(__file__).resolve()
    src_root = here.parents[1]
    backend = get_sampling_backend_name()
    module_path = src_root / backend / str(module_filename)
    spec = spec_from_file_location(str(module_key), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sampling backend module: {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
