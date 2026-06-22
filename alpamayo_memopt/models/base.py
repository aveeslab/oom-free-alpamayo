"""Model adapter interface + shared 3-hook residency pipeline.

Both Alpamayo-R1 and Alpamayo 1.5 share the same memory-optimization
pipeline (paper Section IV): VLM, ViT, and Expert layers are offloaded to
pinned host memory and streamed on demand through three DoubleBufHook
instances that share a pair of GPU buffer slots and a single prefetch CUDA
stream. The residency policy keeps the highest-benefit VLM layers permanently
on the GPU (interleaved placement, paper Eq. 8); ViT and Expert layers, whose
per-inference residency benefit is far lower, stay streamed.

Only the model-specific bits differ between versions and live in a
``ModelAdapter`` subclass:
    - how the model is loaded (class, weights, attn impl, env/CLI knobs),
    - how the VLM / ViT / Expert layer sequences are located,
    - which non-layer modules are made permanently resident,
    - how the benchmark inputs are prepared, and
    - how a single inference is invoked.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Sequence

import torch

from alpamayo_memopt.hook import DoubleBufHook


# ─────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────

def deep_copy_inputs(mi: dict) -> dict:
    """Deep-copy a model-input dict. Tensors are ``.clone()``-d."""
    result: dict = {}
    for k, v in mi.items():
        if isinstance(v, dict):
            result[k] = {
                kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        elif isinstance(v, torch.Tensor):
            result[k] = v.clone()
        else:
            result[k] = copy.deepcopy(v)
    return result


@dataclass
class LoadedModel:
    """Result of ``ModelAdapter.load``: the CPU model + its layer sequences."""
    model: Any
    vlm_layers: Any                       # nn.ModuleList of VLM decoder layers
    vit_blocks: List[Any]                 # ViT transformer blocks
    expert_layers: List[Any]              # diffusion expert layers
    weights_total_gb: float
    settings: dict = field(default_factory=dict)  # adapter-specific, saved to config


# ─────────────────────────────────────────────────────────────────────
# Shared 3-hook pipeline
# ─────────────────────────────────────────────────────────────────────

class TriHookPipeline:
    """VLM + ViT + Expert demand-layering with shared DFB slots + prefetch stream.

    Resident VLM layers must already be on the GPU before constructing the
    pipeline; everything passed via ``vlm_resident`` is kept resident and the
    rest is streamed. ViT and Expert layers are always streamed.
    """

    def __init__(
        self,
        vlm_layers: Sequence[torch.nn.Module],
        vit_blocks: Sequence[torch.nn.Module],
        expert_layers: Sequence[torch.nn.Module],
        vlm_resident: Sequence[int],
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        n_vlm = len(vlm_layers)
        self.vlm_offload = sorted(set(range(n_vlm)) - set(vlm_resident))

        # VLM auto-restarts per decode token; Expert per diffusion step;
        # ViT runs exactly once per inference (no auto-restart).
        self.vlm_hook = DoubleBufHook(auto_restart=True, device=self.device)
        self.vis_hook = DoubleBufHook(auto_restart=False, device=self.device)
        self.exp_hook = DoubleBufHook(auto_restart=True, device=self.device)

        if self.vlm_offload:
            self.vlm_hook.pin(vlm_layers, self.vlm_offload)
        self._vit_idx = list(range(len(vit_blocks)))
        self._exp_idx = list(range(len(expert_layers)))
        self.vis_hook.pin(vit_blocks, self._vit_idx)
        self.exp_hook.pin(expert_layers, self._exp_idx)

        # One pair of GPU buffer slots sized for the largest streamed layer,
        # shared across all three modules (they execute sequentially).
        mx = max(
            self.vlm_hook.max_elements() if self.vlm_offload else 0,
            self.vis_hook.max_elements(),
            self.exp_hook.max_elements(),
        )
        bufs = [
            torch.empty(mx, dtype=torch.bfloat16, device=self.device)
            for _ in range(2)
        ]
        ps = torch.cuda.Stream(device=self.device)
        for h in (self.vlm_hook, self.vis_hook, self.exp_hook):
            h.set_bufs(bufs, prefetch_stream=ps)

        if self.vlm_offload:
            self.vlm_hook.register(vlm_layers, self.vlm_offload)
        self.vis_hook.register(vit_blocks, self._vit_idx)
        self.exp_hook.register(expert_layers, self._exp_idx)

    def start_iteration(self) -> None:
        """Reset DMA state and prime ViT for one inference pass.

        VLM and Expert prime lazily on their first layer's pre-hook and then
        chain via auto-restart, matching the validated reference pipeline.
        """
        self.vlm_hook.reset()
        self.vis_hook.reset()
        self.exp_hook.reset()
        self.vis_hook.start()

    def remove(self) -> None:
        self.vlm_hook.remove()
        self.vis_hook.remove()
        self.exp_hook.remove()


# ─────────────────────────────────────────────────────────────────────
# Adapter interface
# ─────────────────────────────────────────────────────────────────────

class ModelAdapter(ABC):
    """Model-version-specific behavior behind a uniform interface."""

    kind: str = ""
    display_name: str = ""
    default_model_id: str = ""
    default_max_generation_length: int = 256
    default_num_traj_samples: int = 1
    # Representative decode-token count (used only for reporting / config).
    decode_tokens: int = 21

    def add_args(self, parser) -> None:  # noqa: D401
        """Register adapter-specific CLI flags. Override as needed."""
        return None

    @abstractmethod
    def load(self, args, config=None) -> LoadedModel:
        """Load the model on CPU and locate its VLM/ViT/Expert layers.

        ``config`` (when given, at inference time) lets an adapter fall back to
        values saved during profiling for any knob not set on the CLI/env.
        """

    @abstractmethod
    def setup_essentials(self, loaded: LoadedModel, device: str) -> None:
        """Move always-resident non-layer modules to ``device``.

        Must leave the VLM / ViT / Expert layer sequences on CPU so the hooks
        can stream them.
        """

    @abstractmethod
    def prepare_inputs(self, loaded: LoadedModel, args, device: str):
        """Build the benchmark inputs and move them to ``device``."""

    @abstractmethod
    def run(self, loaded: LoadedModel, inputs, args):
        """Invoke one inference pass and return its output."""

    def extra_model_config(self, loaded: LoadedModel, args) -> dict:
        """Adapter-specific fields merged into the saved ModelConfig."""
        return {}
