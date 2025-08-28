# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
from functools import partial

import torch
from action_head_utils import action_head_pytorch_forward
from trt_model_forward import setup_tensorrt_engines

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset, LE_ROBOT_MODALITY_FILENAME
from gr00t.experiment.data_config import get_data_config, DATA_CONFIG_MAP, DYNAMIC_DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.utils.misc import read_json


def compare_predictions(pred_tensorrt, pred_torch):
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len("Cosine Similarity (PyTorch/TensorRT):"),
        len("L1 Mean/Max Distance (PyTorch/TensorRT):"),
        len("Max Output Values (PyTorch/TensorRT):"),
        len("Mean Output Values (PyTorch/TensorRT):"),
        len("Min Output Values (PyTorch/TensorRT):"),
    )

    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"{key} shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(f"\n{key}:")
        print(
            f'{"Cosine Similarity (PyTorch/TensorRT):".ljust(max_label_width)} {cos_sim.item()}')
        print(
            f'{"L1 Mean/Max Distance (PyTorch/TensorRT):".ljust(max_label_width)} {l1_dist.mean().item():.4f}/{l1_dist.max().item():.4f}'
        )
        print(
            f'{"Max Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.max().item():.4f}/{tensorrt_tensor.max().item():.4f}'
        )
        print(
            f'{"Mean Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.mean().item():.4f}/{tensorrt_tensor.mean().item():.4f}'
        )
        print(
            f'{"Min Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.min().item():.4f}/{tensorrt_tensor.min().item():.4f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GR00T inference")

    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default=os.path.join(os.getcwd(), "demo_data/robot_sim.PickNPlace"),
    )

    parser.add_argument(
        "--model_path", type=str, default="nvidia/GR00T-N1.5-3B", help="Path to the GR00T model"
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["pytorch", "tensorrt", "compare"],
        default="pytorch",
        help="Inference mode: 'pytorch' for PyTorch inference, 'tensorrt' for TensorRT inference, 'compare' for compare PyTorch and TensorRT outputs similarity",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps",
        default=4,
    )

    parser.add_argument(
        "--data_config",
        type=str,
        choices=list(DATA_CONFIG_MAP.keys()) +
        list(DYNAMIC_DATA_CONFIG_MAP.keys()),
        help=f"Data configuration name. Available options: {', '.join(list(DATA_CONFIG_MAP.keys()) + list(DYNAMIC_DATA_CONFIG_MAP.keys()))}",
        default="extend_robotics_dynamic",
    )

    parser.add_argument(
        "--action_dim",
        type=int,
        default=32,
        help="Dimensionality of the action space",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help="Size of each data chunk",
    )

    parser.add_argument(
        "--video_backend",
        type=str,
        choices=["torchvision_av", "decord"],
        default="torchvision_av",
        help="Backend for video processing",
    )

    parser.add_argument(
        "--trt_engine_path",
        type=str,
        help="Path to the TensorRT engine",
        default="gr00t_engine",
    )
    args = parser.parse_args()

    modality_path = os.path.join(args.dataset_path, LE_ROBOT_MODALITY_FILENAME)
    modality_dict = read_json(modality_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config_obj = get_data_config(name=args.data_config,
                                      modality_map=modality_dict,
                                      chunk_size=args.chunk_size,
                                      action_dim=args.action_dim
                                      )
    modality_config = data_config_obj.modality_config()
    modality_transform = data_config_obj.transform()

    if args.data_config in (
        "fourier_gr1_arms_waist",
        "fourier_gr1_arms_only",
        "fourier_gr1_full_upper_body",
    ):
        EMBODIMENT_TAG = "gr1"
    elif args.data_config == "oxe_droid":
        EMBODIMENT_TAG = "oxe_droid"
    elif args.data_config == "agibot_genie1":
        EMBODIMENT_TAG = "agibot_genie1"
    else:
        EMBODIMENT_TAG = "new_embodiment"

    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device=device,
    )

    modality_config = policy.modality_config
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )

    step_data = dataset[0]

    if args.inference_mode == "pytorch":
        predicted_action = policy.get_action(step_data)
        print("\n=== PyTorch Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)

    elif args.inference_mode == "tensorrt":
        # Setup TensorRT engines
        setup_tensorrt_engines(policy, args.trt_engine_path)

        predicted_action = policy.get_action(step_data)
        print("\n=== TensorRT Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)

    else:
        # ensure PyTorch and TensorRT have the same init_actions
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.randn(
                (1, policy.model.action_head.action_horizon,
                 policy.model.action_head.action_dim),
                dtype=torch.float16,
                device=device,
            )
        # PyTorch inference
        policy.model.action_head.get_action = partial(
            action_head_pytorch_forward, policy.model.action_head
        )
        predicted_action_torch = policy.get_action(step_data)

        # Setup TensorRT engines and run inference
        setup_tensorrt_engines(policy, args.trt_engine_path)
        predicted_action_tensorrt = policy.get_action(step_data)

        # Compare predictions
        compare_predictions(predicted_action_tensorrt, predicted_action_torch)
