"""alpamayo15-memory-optimizer

Memory-efficient inference adapter for NVIDIA Alpamayo 1.5 Vision-Language-Action
model on resource-constrained GPU platforms.

Main components:
    - DoubleBufHook: Pipelined Demand Layering with Double Flat Buffer
    - profiler:     System spec detection + per-layer profiling + config generation
    - config:       Config JSON serialization
"""

__version__ = "0.1.0"

from alpamayo_memopt.hook import DoubleBufHook
from alpamayo_memopt.config import Config, load_config, save_config

__all__ = [
    "DoubleBufHook",
    "Config",
    "load_config",
    "save_config",
]
