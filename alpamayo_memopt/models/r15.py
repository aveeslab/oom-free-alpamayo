"""Alpamayo 1.5 adapter.

Keeps the optimizer decoupled from the NVIDIA Alpamayo 1.5 source tree: the
source path is added to ``sys.path`` at runtime and weights are loaded through
HuggingFace/Transformers — never copied into this repo. Loading knobs resolve
with precedence CLI flag > environment variable > saved config > fallback, so
``python scripts/infer.py --config config.json`` works once the environment is
set, while every value remains overridable per run.

The memory-optimization pipeline is shared (see ``TriHookPipeline``): ViT and
Expert layers are streamed alongside the offloaded VLM layers, so this adapter
deliberately leaves them on CPU during ``setup_essentials``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from alpamayo_memopt.models.base import LoadedModel, ModelAdapter
from alpamayo_memopt.profiler import get_model_weight_size_gb


# Environment variable names.
ENV_SRC = "ALPAMAYO15_SRC"
ENV_MODEL_ID = "ALPAMAYO15_MODEL_ID"
ENV_MODEL_CACHE_DIR = "ALPAMAYO15_MODEL_CACHE_DIR"
ENV_MODEL_REVISION = "ALPAMAYO15_MODEL_REVISION"
ENV_LOCAL_FILES_ONLY = "ALPAMAYO15_LOCAL_FILES_ONLY"
ENV_ATTN = "ALPAMAYO15_ATTN_IMPLEMENTATION"
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


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _ensure_importable(src: str | Path | None):
    """Add the Alpamayo 1.5 source path to sys.path and verify the import."""
    if src:
        src = Path(src).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Alpamayo 1.5 source path not found: {src}")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    try:
        import alpamayo1_5  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Could not import alpamayo1_5. Pass --alpamayo-src, set "
            f"{ENV_SRC}, or install the Alpamayo 1.5 package."
        ) from e


class R15Adapter(ModelAdapter):
    kind = "r15"
    display_name = "Alpamayo 1.5"
    default_model_id = FALLBACK_MODEL_ID
    default_max_generation_length = 256
    default_num_traj_samples = 1
    decode_tokens = 21

    # ── CLI ───────────────────────────────────────────────────────────
    def add_args(self, parser) -> None:
        g = parser.add_argument_group("Alpamayo 1.5 model options")
        g.add_argument("--alpamayo-src", type=Path, default=None,
                       help=f"Path to Alpamayo 1.5 src. Defaults to ${ENV_SRC}, "
                            "then config, then an installed alpamayo1_5 package.")
        g.add_argument("--model-id", default=None,
                       help=f"Model id/path. Defaults to ${ENV_MODEL_ID} / config / "
                            f"{FALLBACK_MODEL_ID}.")
        g.add_argument("--model-cache-dir", type=Path, default=None,
                       help="Optional HuggingFace cache_dir.")
        g.add_argument("--model-revision", default=None,
                       help="Optional model revision for from_pretrained.")
        g.add_argument("--local-files-only", action="store_true", default=None,
                       help="Pass local_files_only=True to from_pretrained.")
        g.add_argument("--attn-implementation", default=None,
                       help="Transformers attention implementation (e.g. eager).")
        g.add_argument("--clip-id", default=None,
                       help="physical_ai_av clip id for the benchmark input.")
        g.add_argument("--t0-us", type=int, default=None,
                       help="physical_ai_av timestamp for the benchmark input.")
        g.add_argument("--dataset-revision", dest="dataset_revisions",
                       action="append", default=None,
                       help="physical_ai_av revision candidate (repeatable / CSV).")

    # ── Precedence resolution (CLI > env > config > fallback) ──────────
    def _resolve(self, args, config) -> dict:
        m = config.model if config is not None else None

        def pick(cli, env_name, cfg_value, fallback=None):
            if cli is not None and cli != "":
                return cli
            env = os.environ.get(env_name)
            if env:
                return env
            if cfg_value:
                return cfg_value
            return fallback

        src = pick(getattr(args, "alpamayo_src", None), ENV_SRC,
                   m.alpamayo_source_path if m else None)
        cache = pick(getattr(args, "model_cache_dir", None), ENV_MODEL_CACHE_DIR,
                     m.model_cache_dir if m else None)
        local = getattr(args, "local_files_only", None)
        if local is None:
            local = _env_bool(ENV_LOCAL_FILES_ONLY)
        if local is None:
            local = m.local_files_only if m else False

        if getattr(args, "dataset_revisions", None):
            revs = tuple(r for item in args.dataset_revisions for r in _split(item))
        elif os.environ.get(ENV_DATASET_REVISIONS) or os.environ.get(ENV_DATASET_REVISION):
            revs = tuple(_split(os.environ.get(ENV_DATASET_REVISIONS))
                         + _split(os.environ.get(ENV_DATASET_REVISION)))
        elif m and m.dataset_revisions:
            revs = tuple(m.dataset_revisions)
        else:
            revs = FALLBACK_DATASET_REVISIONS

        t0 = getattr(args, "t0_us", None)
        if t0 is None:
            t0 = int(os.environ.get(ENV_T0_US) or (m.t0_us if m and m.t0_us else FALLBACK_T0_US))

        return {
            "src": str(Path(src).expanduser()) if src else None,
            "model_id": pick(getattr(args, "model_id", None), ENV_MODEL_ID,
                             m.name if m else None, FALLBACK_MODEL_ID),
            "cache_dir": str(Path(cache).expanduser()) if cache else None,
            "revision": pick(getattr(args, "model_revision", None),
                             ENV_MODEL_REVISION, m.model_revision if m else None),
            "local_files_only": bool(local),
            "attn": pick(getattr(args, "attn_implementation", None), ENV_ATTN,
                         m.attn_implementation if m else None),
            "clip_id": pick(getattr(args, "clip_id", None), ENV_CLIP_ID,
                            m.clip_id if m else None, FALLBACK_CLIP_ID),
            "t0_us": int(t0),
            "dataset_revisions": list(revs),
        }

    # ── Loading ───────────────────────────────────────────────────────
    def load(self, args, config=None) -> LoadedModel:
        s = self._resolve(args, config)
        _ensure_importable(s["src"])
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
        }
        if s["attn"]:
            kwargs["attn_implementation"] = s["attn"]
        if s["cache_dir"]:
            kwargs["cache_dir"] = s["cache_dir"]
        if s["revision"]:
            kwargs["revision"] = s["revision"]
        if s["local_files_only"]:
            kwargs["local_files_only"] = True

        model = Alpamayo1_5.from_pretrained(s["model_id"], **kwargs)
        model.eval()

        lm = model.vlm.model.language_model
        visual = model.vlm.model.visual
        return LoadedModel(
            model=model,
            vlm_layers=lm.layers,
            vit_blocks=list(visual.blocks),
            expert_layers=list(model.expert.layers),
            weights_total_gb=get_model_weight_size_gb(model),
            settings=s,
        )

    # ── GPU placement of always-resident essentials ───────────────────
    def setup_essentials(self, loaded: LoadedModel, device: str) -> None:
        model = loaded.model
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

        # Expert: keep everything except the layer stack resident.
        for name, child in model.expert.named_children():
            if name != "layers":
                child.to(device)
        for name, p in model.expert.named_parameters():
            if "layers." not in name and p.device.type == "cpu":
                p.data = p.data.to(device)
        for name, b in model.expert.named_buffers():
            if "layers." not in name and b.device.type == "cpu":
                b.data = b.data.to(device)

    # ── Benchmark inputs ──────────────────────────────────────────────
    def prepare_inputs(self, loaded: LoadedModel, args, device: str):
        s = loaded.settings
        _ensure_importable(s.get("src"))
        import physical_ai_av
        from alpamayo1_5 import helper
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

        avdi = None
        for rev in (s.get("dataset_revisions") or FALLBACK_DATASET_REVISIONS):
            try:
                avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=rev)
                break
            except Exception:
                continue
        if avdi is None:
            avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

        data = load_physical_aiavdataset(s["clip_id"], t0_us=s["t0_us"], avdi=avdi)
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        processor = helper.get_processor(loaded.model.tokenizer)
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        return helper.to_device({
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }, device)

    # ── Inference ─────────────────────────────────────────────────────
    def run(self, loaded: LoadedModel, inputs, args):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return loaded.model.sample_trajectories_from_data_with_vlm_rollout(
                data=inputs,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
            )

    # ── Config persistence ────────────────────────────────────────────
    def extra_model_config(self, loaded: LoadedModel, args) -> dict:
        s = loaded.settings
        return {
            "alpamayo_source_path": s.get("src") or "",
            "model_cache_dir": s.get("cache_dir") or "",
            "model_revision": s.get("revision") or "",
            "attn_implementation": s.get("attn") or "",
            "local_files_only": bool(s.get("local_files_only")),
            "clip_id": s.get("clip_id") or "",
            "t0_us": int(s.get("t0_us") or 0),
            "dataset_revisions": list(s.get("dataset_revisions") or []),
        }
