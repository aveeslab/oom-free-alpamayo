"""oom-free-alpamayo

Memory-efficient inference framework for NVIDIA Alpamayo Vision-Language-Action
models (Alpamayo-R1 and Alpamayo 1.5) on resource-constrained GPU platforms.

Main components:
    - DoubleBufHook:   Pipelined Demand Layering with Double Flat Buffer
    - models:          per-version adapters (get_adapter) + shared TriHookPipeline
    - profiler:        system / model / bandwidth profiling primitives
    - config:          Config JSON serialization
    - gpu:             optional GPU graphics-clock locking
"""

__version__ = "0.2.0"

from alpamayo_memopt.hook import DoubleBufHook
from alpamayo_memopt.config import Config, load_config, save_config
from alpamayo_memopt.models import (
    ADAPTER_CHOICES,
    DEFAULT_KIND,
    ModelAdapter,
    TriHookPipeline,
    get_adapter,
)

__all__ = [
    "DoubleBufHook",
    "Config",
    "load_config",
    "save_config",
    "ModelAdapter",
    "TriHookPipeline",
    "get_adapter",
    "ADAPTER_CHOICES",
    "DEFAULT_KIND",
]
