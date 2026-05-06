"""Alpamayo 1.5 integration helpers.

This module keeps the memory optimizer separate from the NVIDIA Alpamayo 1.5
source tree. The source path is added to sys.path at runtime; model weights are
loaded through HuggingFace/Transformers and are never copied into this repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ENV_ALPAMAYO15_SRC = "ALPAMAYO15_SRC"
ENV_MODEL_ID = "ALPAMAYO15_MODEL_ID"
ENV_MODEL_CACHE_DIR = "ALPAMAYO15_MODEL_CACHE_DIR"
ENV_MODEL_REVISION = "ALPAMAYO15_MODEL_REVISION"
ENV_LOCAL_FILES_ONLY = "ALPAMAYO15_LOCAL_FILES_ONLY"
ENV_ATTN_IMPLEMENTATION = "ALPAMAYO15_ATTN_IMPLEMENTATION"
ENV_CLIP_ID = "ALPAMAYO15_CLIP_ID"
ENV_T0_US = "ALPAMAYO15_T0_US"
ENV_DATASET_REVISION = "ALPAMAYO15_DATASET_REVISION"
ENV_DATASET_REVISIONS = "ALPAMAYO15_DATASET_REVISIONS"

FALLBACK_MODEL_ID = "nvidia/Alpamayo-1.5-10B"
FALLBACK_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
FALLBACK_T0_US = 5_100_000
FALLBACK_DATASET_REVISIONS = (
    "2ae73f49ffd2b5db43b404201beb7b92889f7afc",
    "37a7cc2c868d684d0456b5412a7ec5d18597a96a",
)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


DEFAULT_ALPAMAYO15_SRC = _env_path(ENV_ALPAMAYO15_SRC)
DEFAULT_MODEL_ID = os.environ.get(ENV_MODEL_ID, FALLBACK_MODEL_ID)
DEFAULT_MODEL_CACHE_DIR = _env_path(ENV_MODEL_CACHE_DIR)
DEFAULT_MODEL_REVISION = os.environ.get(ENV_MODEL_REVISION)
DEFAULT_LOCAL_FILES_ONLY = _env_bool(ENV_LOCAL_FILES_ONLY)
DEFAULT_ATTN_IMPLEMENTATION = os.environ.get(ENV_ATTN_IMPLEMENTATION)
DEFAULT_CLIP_ID = os.environ.get(ENV_CLIP_ID, FALLBACK_CLIP_ID)
DEFAULT_T0_US = _env_int(ENV_T0_US, FALLBACK_T0_US)


def resolve_alpamayo15_source_path(
    source_path: str | Path | None = None,
    config_source_path: str | Path | None = None,
) -> Path | None:
    """Resolve source path with precedence: CLI value, env var, config value."""
    for candidate in (source_path, DEFAULT_ALPAMAYO15_SRC, config_source_path):
        if candidate:
            return Path(candidate).expanduser()
    return None


def resolve_dataset_revisions(
    revisions: Sequence[str] | str | None = None,
) -> tuple[str, ...]:
    """Resolve physical_ai_av revision candidates from CLI/env/defaults."""
    values: list[str] = []
    if revisions is None:
        values.extend(_split_values(os.environ.get(ENV_DATASET_REVISIONS)))
        values.extend(_split_values(os.environ.get(ENV_DATASET_REVISION)))
    elif isinstance(revisions, str):
        values.extend(_split_values(revisions))
    else:
        for revision in revisions:
            values.extend(_split_values(revision))

    if not values:
        return FALLBACK_DATASET_REVISIONS
    return tuple(values)


def ensure_alpamayo15_importable(source_path: str | Path | None = None) -> Path | None:
    """Add the Alpamayo 1.5 source path to sys.path and verify imports."""
    src = resolve_alpamayo15_source_path(source_path)
    if src is not None:
        src = src.resolve()
        if not src.exists():
            raise FileNotFoundError(f"Alpamayo 1.5 source path not found: {src}")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

    try:
        import alpamayo1_5
    except ImportError as e:
        raise ImportError(
            "Could not import alpamayo1_5. Pass --alpamayo-src, set "
            f"{ENV_ALPAMAYO15_SRC}, or install the Alpamayo 1.5 package in "
            "the active environment."
        ) from e

    module_file = getattr(alpamayo1_5, "__file__", None)
    if module_file:
        return Path(module_file).resolve().parent.parent
    return src


def load_alpamayo15(
    source_path: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    attn_implementation: str | None = None,
    cache_dir: str | Path | None = DEFAULT_MODEL_CACHE_DIR,
    revision: str | None = DEFAULT_MODEL_REVISION,
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
):
    """Load Alpamayo 1.5 on CPU and return model plus layer sequences."""
    ensure_alpamayo15_importable(source_path)
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    if revision is not None:
        kwargs["revision"] = revision
    if local_files_only:
        kwargs["local_files_only"] = True

    model = Alpamayo1_5.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, get_vlm_layers(model), get_expert_layers(model)


def get_vlm_layers(model) -> torch.nn.ModuleList:
    """Return Qwen3-VL language-model transformer layers."""
    return model.vlm.model.language_model.layers


def get_expert_layers(model) -> torch.nn.ModuleList:
    """Return Alpamayo diffusion expert transformer layers."""
    layers = getattr(model.expert, "layers", None)
    if layers is None:
        raise AttributeError("Expected model.expert.layers on Alpamayo 1.5")
    return layers


def setup_non_layer_components(model, device: str = "cuda") -> None:
    """Move non-VLM modules to GPU while leaving only VLM layers on CPU."""
    lm = model.vlm.model.language_model
    visual = model.vlm.model.visual

    lm.embed_tokens.to(device)
    lm.norm.to(device)
    lm.rotary_emb.to(device)
    model.vlm.lm_head.to(device)

    for attr in ("action_space", "action_in_proj", "action_out_proj", "diffusion"):
        if hasattr(model, attr):
            getattr(model, attr).to(device)

    for name, module in visual.named_children():
        if name != "blocks":
            module.to(device)
    for block in visual.blocks:
        block.to(device)

    model.expert.to(device)


def prepare_default_inputs(
    model,
    source_path: str | Path | None = None,
    device: str = "cuda",
    clip_id: str = DEFAULT_CLIP_ID,
    t0_us: int = DEFAULT_T0_US,
    dataset_revisions: Sequence[str] | str | None = None,
):
    """Prepare the Alpamayo 1.5 release example input."""
    ensure_alpamayo15_importable(source_path)
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    import physical_ai_av

    avdi = None
    for rev in resolve_dataset_revisions(dataset_revisions):
        try:
            avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=rev)
            break
        except Exception:
            continue
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, device)


