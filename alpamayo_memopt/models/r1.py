"""Alpamayo-R1-10B adapter.

Ports the validated R1 setup helpers (model loading, GPU placement of
non-layer essentials, benchmark-input preparation, and the inference call)
behind the shared ``ModelAdapter`` interface. The memory-optimization
pipeline itself is shared (see ``alpamayo_memopt.models.base.TriHookPipeline``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from alpamayo_memopt.models.base import LoadedModel, ModelAdapter, deep_copy_inputs
from alpamayo_memopt.profiler import get_model_weight_size_gb


# Constants (keep in sync with the paper).
MODEL_ID = "nvidia/Alpamayo-R1-10B"
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
HF_CACHED_REVISION = "2ae73f49ffd2b5db43b404201beb7b92889f7afc"
DATASET_FALLBACK_REVISION = "37a7cc2c868d684d0456b5412a7ec5d18597a96a"
N_VLM = 36
DECODE_TOKENS = 21


class R1Adapter(ModelAdapter):
    kind = "r1"
    display_name = "Alpamayo-R1-10B"
    default_model_id = MODEL_ID
    default_max_generation_length = 22
    default_num_traj_samples = 1
    decode_tokens = DECODE_TOKENS

    # ── CLI ───────────────────────────────────────────────────────────
    def add_args(self, parser) -> None:
        g = parser.add_argument_group("Alpamayo-R1 model options")
        g.add_argument("--model-id", default=MODEL_ID,
                       help=f"Model id/path for from_pretrained (default: {MODEL_ID}).")
        g.add_argument("--model-cache-dir", type=Path, default=None,
                       help="Optional HuggingFace cache_dir.")
        g.add_argument("--model-revision", default=None,
                       help="Optional model revision for from_pretrained.")
        g.add_argument("--attn-implementation", default=None,
                       help="Optional Transformers attention implementation (e.g. eager).")
        g.add_argument("--local-files-only", action="store_true",
                       help="Pass local_files_only=True to from_pretrained.")

    # ── Loading ───────────────────────────────────────────────────────
    def load(self, args, config=None) -> LoadedModel:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

        kwargs: dict = {
            "dtype": torch.bfloat16,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
        }
        if getattr(args, "attn_implementation", None):
            kwargs["attn_implementation"] = args.attn_implementation
        if getattr(args, "model_cache_dir", None):
            kwargs["cache_dir"] = str(Path(args.model_cache_dir).expanduser())
        if getattr(args, "model_revision", None):
            kwargs["revision"] = args.model_revision
        if getattr(args, "local_files_only", False):
            kwargs["local_files_only"] = True

        model_id = getattr(args, "model_id", None) or MODEL_ID
        model = AlpamayoR1.from_pretrained(model_id, **kwargs)
        model.eval()

        lm = model.vlm.model.language_model
        visual = model.vlm.model.visual
        return LoadedModel(
            model=model,
            vlm_layers=lm.layers,
            vit_blocks=list(visual.blocks),
            expert_layers=list(model.expert.layers),
            weights_total_gb=get_model_weight_size_gb(model),
            settings={"model_id": model_id},
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
        for attr in ("action_in_proj", "action_out_proj", "diffusion"):
            if hasattr(model, attr):
                getattr(model, attr).to(device)

        # Vision: everything except the transformer blocks stays resident.
        for name, module in visual.named_children():
            if name != "blocks":
                module.to(device)

        # Expert: everything except the layer stack stays resident.
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
        import physical_ai_av
        from alpamayo_r1 import helper
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

        avdi = None
        for rev in (HF_CACHED_REVISION, DATASET_FALLBACK_REVISION):
            try:
                avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=rev)
                break
            except Exception:
                continue
        if avdi is None:
            avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

        data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US, avdi=avdi)
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
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
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return loaded.model.sample_trajectories_from_data_with_vlm_rollout(
                data=deep_copy_inputs(inputs),
                top_p=1.0, temperature=0.0,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
                return_extra=True,
            )
