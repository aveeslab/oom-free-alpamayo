"""Config JSON serialization.

Schema:
{
  "system": {
    "gpu_name": str,
    "vram_total_gb": float,
    "vram_budget_gb": float,
    "cpu_dram_total_gb": float
  },
  "model": {
    "name": str,
    "weights_total_gb": float,
    "vlm_layers": int,
    "vlm_layer_size_mb": float,
    "expert_layers": int,
    "expert_layer_size_mb": float,
    "alpamayo_source_path": str,
    "model_cache_dir": str,
    "model_revision": str,
    "attn_implementation": str,
    "local_files_only": bool,
    "clip_id": str,
    "t0_us": int,
    "dataset_revisions": [str, ...]
  },
  "profiling": {
    "vlm_layer_dma_ms": float,
    "vlm_layer_exe_ms": float,
    "vit_layer_dma_ms": float,
    "vit_layer_exe_ms": float,
    "non_vlm_overhead_gb": float
  },
  "residency": {
    "max_possible": int,
    "num_resident": int,
    "resident_indices": [int, ...],
    "expert_max_possible": int,
    "expert_num_resident": int,
    "expert_resident_indices": [int, ...],
    "placement_rule": "interleaved (paper Eq. interleaved_placement)"
  },
  "_meta": {
    "alpamayo_memopt_version": str,
    "created_at": str  # ISO 8601
  }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class SystemConfig:
    gpu_name: str
    vram_total_gb: float
    vram_budget_gb: float
    cpu_dram_total_gb: float


@dataclass
class ModelConfig:
    name: str
    weights_total_gb: float
    vlm_layers: int
    vlm_layer_size_mb: float
    kind: str = "r1"  # adapter that produced this config: "r1" | "r15"
    expert_layers: int = 0
    expert_layer_size_mb: float = 0.0
    alpamayo_source_path: str = ""
    model_cache_dir: str = ""
    model_revision: str = ""
    attn_implementation: str = ""
    local_files_only: bool = False
    clip_id: str = ""
    t0_us: int = 0
    dataset_revisions: List[str] = field(default_factory=list)


@dataclass
class ProfilingConfig:
    vlm_layer_dma_ms: float
    vlm_layer_exe_ms: float
    vit_layer_dma_ms: float
    vit_layer_exe_ms: float
    non_vlm_overhead_gb: float


@dataclass
class ResidencyConfig:
    max_possible: int
    num_resident: int
    resident_indices: List[int]
    expert_max_possible: int = 0
    expert_num_resident: int = 0
    expert_resident_indices: List[int] = field(default_factory=list)
    placement_rule: str = "interleaved (paper Eq. interleaved_placement)"


@dataclass
class Config:
    system: SystemConfig
    model: ModelConfig
    profiling: ProfilingConfig
    residency: ResidencyConfig
    _meta: dict = field(default_factory=dict)


def save_config(config: Config, path: str | Path) -> None:
    """Serialize config to JSON file."""
    from alpamayo_memopt import __version__

    data = asdict(config)
    data["_meta"] = {
        "alpamayo_memopt_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_config(path: str | Path) -> Config:
    """Load config from JSON file."""
    data = json.loads(Path(path).read_text())

    def _filter(cls, values: dict) -> dict:
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in values.items() if k in allowed}

    return Config(
        system=SystemConfig(**_filter(SystemConfig, data["system"])),
        model=ModelConfig(**_filter(ModelConfig, data["model"])),
        profiling=ProfilingConfig(**_filter(ProfilingConfig, data["profiling"])),
        residency=ResidencyConfig(**_filter(ResidencyConfig, data["residency"])),
        _meta=data.get("_meta", {}),
    )
