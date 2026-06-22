# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end baseline script for the original Alpamayo 1.5 inference path.

Loads a dataset, runs full-GPU inference, and computes the minADE.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from alpamayo_memopt import alpamayo15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline Alpamayo 1.5 inference.")
    p.add_argument("--alpamayo-src", type=Path, default=None,
                   help=("Path to Alpamayo 1.5 src directory. Defaults to "
                         "ALPAMAYO15_SRC, then an installed alpamayo1_5 package."))
    p.add_argument("--model-id", default=alpamayo15.DEFAULT_MODEL_ID,
                   help=("Model id/path passed to from_pretrained. Defaults to "
                         "ALPAMAYO15_MODEL_ID."))
    p.add_argument("--model-cache-dir", type=Path, default=alpamayo15.DEFAULT_MODEL_CACHE_DIR,
                   help=("Optional HuggingFace/Transformers cache_dir. Defaults to "
                         "ALPAMAYO15_MODEL_CACHE_DIR."))
    p.add_argument("--model-revision", default=alpamayo15.DEFAULT_MODEL_REVISION,
                   help=("Optional model revision passed to from_pretrained. Defaults "
                         "to ALPAMAYO15_MODEL_REVISION."))
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction,
                   default=alpamayo15.DEFAULT_LOCAL_FILES_ONLY,
                   help=("Pass local_files_only to from_pretrained. Defaults to "
                         "ALPAMAYO15_LOCAL_FILES_ONLY."))
    p.add_argument("--attn-implementation", default=alpamayo15.DEFAULT_ATTN_IMPLEMENTATION,
                   help="Optional Transformers attention implementation override, e.g. eager.")
    p.add_argument("--clip-id", default=alpamayo15.DEFAULT_CLIP_ID,
                   help="physical_ai_av clip id. Defaults to ALPAMAYO15_CLIP_ID.")
    p.add_argument("--t0-us", type=int, default=alpamayo15.DEFAULT_T0_US,
                   help="physical_ai_av timestamp. Defaults to ALPAMAYO15_T0_US.")
    p.add_argument("--dataset-revision", dest="dataset_revisions", action="append",
                   default=None,
                   help=("physical_ai_av revision candidate. Can be repeated or "
                         "comma-separated. Defaults to ALPAMAYO15_DATASET_REVISIONS."))
    p.add_argument("--device", default="cuda", help="Torch device for baseline inference.")
    return p.parse_args()


def main() -> None:
    """Run inference on an example clip and report minADE."""
    args = parse_args()
    alpamayo15.ensure_alpamayo15_importable(args.alpamayo_src)

    import physical_ai_av
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    clip_id = args.clip_id
    print(f"Loading dataset for clip_id: {clip_id}...")
    avdi = None
    for rev in alpamayo15.resolve_dataset_revisions(args.dataset_revisions):
        try:
            avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=rev)
            break
        except Exception:
            continue
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    data = load_physical_aiavdataset(clip_id, t0_us=args.t0_us, avdi=avdi)
    print("Dataset loaded.")
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )

    model_kwargs = {"dtype": torch.bfloat16}
    if args.model_cache_dir is not None:
        model_kwargs["cache_dir"] = str(args.model_cache_dir.expanduser())
    if args.model_revision is not None:
        model_kwargs["revision"] = args.model_revision
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    # model = Alpamayo1_5.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map = "auto", max_memory={0: "12GiB", "cpu": "64GiB"})
    model = Alpamayo1_5.from_pretrained(args.model_id, **model_kwargs).to(args.device)
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

    model_inputs = helper.to_device(model_inputs, args.device)

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print("minADE:", min_ade, "meters")
    if min_ade >= 1.0:
        print(f"WARNING: minADE ({min_ade:.2f}m) is above 1.0m. Model sampling can be stochastic.")


if __name__ == "__main__":
    main()
