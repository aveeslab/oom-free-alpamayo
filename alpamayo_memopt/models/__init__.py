"""Model adapter registry.

Each Alpamayo version is wrapped in a :class:`ModelAdapter` that supplies the
model-specific loading / placement / input / inference details. The shared
memory-optimization pipeline lives in :mod:`alpamayo_memopt.models.base`.
"""

from __future__ import annotations

from alpamayo_memopt.models.base import LoadedModel, ModelAdapter, TriHookPipeline
from alpamayo_memopt.models.r1 import R1Adapter
from alpamayo_memopt.models.r15 import R15Adapter

# Insertion order doubles as the CLI choice order.
_ADAPTERS = {
    "r15": R15Adapter,
    "r1": R1Adapter,
}

# Default model for the CLI when --model is omitted.
DEFAULT_KIND = "r15"

ADAPTER_CHOICES = list(_ADAPTERS)


def get_adapter(kind: str) -> ModelAdapter:
    """Return a fresh adapter instance for ``kind`` (e.g. "r1", "r15")."""
    try:
        return _ADAPTERS[kind]()
    except KeyError:
        raise ValueError(
            f"Unknown model '{kind}'. Choose one of: {', '.join(_ADAPTERS)}."
        ) from None


__all__ = [
    "LoadedModel",
    "ModelAdapter",
    "TriHookPipeline",
    "get_adapter",
    "ADAPTER_CHOICES",
    "DEFAULT_KIND",
]
