"""Sampling backend loader for point-mapping modules.

Backends:
- ``pure_angular`` (default)
- ``3region_mapping`` (under ``backup/sampling/``; set ``ISO_SWARM_SAMPLING_BACKEND``)

Set env ``ISO_SWARM_SAMPLING_BACKEND`` to switch.
"""

from __future__ import annotations

import os
import sys
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
    src_root = here.parents[2]
    backend = get_sampling_backend_name()
    if backend == "3region_mapping":
        module_path = src_root / "backup" / "sampling" / "3region_mapping" / str(module_filename)
    else:
        module_path = src_root / "functions" / backend / str(module_filename)
    spec = spec_from_file_location(str(module_key), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sampling backend module: {module_path}")
    module = module_from_spec(spec)
    # Required before exec_module: @dataclass and other introspection use cls.__module__
    # in sys.modules (fails with AttributeError if the loader key is missing).
    sys.modules[str(module_key)] = module
    spec.loader.exec_module(module)
    return module
