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

#!/bin/bash

set -e

# -------------------- Defaults --------------------
NUM_CAMERAS=1
ACTION_DIM=32
ONNX_MODEL_PATH="gr00t_onnx"
OUTPUT_ENGINE_PATH="$HOME/.cache/gr00t-engine-$(date +%Y%m%d-%H%M%S)"

# -------------------- Parse kwargs --------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-cameras)
      NUM_CAMERAS="$2"
      shift 2
      ;;
    --action-dim)
      ACTION_DIM="$2"
      shift 2
      ;;
    --onnx-model-path)
      ONNX_MODEL_PATH="$2"
      shift 2
      ;;
    --output-engine-path)
      OUTPUT_ENGINE_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--num-cameras N] [--action-dim D] [--onnx-model-path DIR] [--output-engine-path PATH]"
      exit 1
      ;;
  esac
done

# -------------------- Derived values --------------------
MIN_LEN=80
OPT_LEN=283
MAX_LEN=$((NUM_CAMERAS * 300))
VIT_SEQ_LEN=$((NUM_CAMERAS * 256))

# -------------------- Info --------------------
echo "Using parameters:"
echo "  num_cameras = ${NUM_CAMERAS}"
echo "  action_dim  = ${ACTION_DIM}"
echo "  output_engine_path = ${OUTPUT_ENGINE_PATH}"
echo "  min_len     = ${MIN_LEN}"
echo "  opt_len     = ${OPT_LEN}"
echo "  max_len     = ${MAX_LEN}"
echo "  vit_seq_len = ${VIT_SEQ_LEN}  (num_cameras * 256)"

echo "Important Notes:"
echo "1: The max batch of engine size is set to 8 in the reference case."
echo "2: The MIN_LEN/OPT_LEN/MAX_LEN for LLM, DiT, VLLN-VLSelfAttention models is set to ${MIN_LEN}/${OPT_LEN}/${MAX_LEN}."
echo "   Adjust these for your actual batch size and sequence lengths."

export PATH=/usr/src/tensorrt/bin:$PATH

if [ ! -e /usr/src/tensorrt/bin/trtexec ]; then
    echo "The file /usr/src/tensorrt/bin/trtexec does not exist. Please install TensorRT"
    exit 1
fi

if [ ! -d "${ONNX_MODEL_PATH}" ]; then
    echo "The provided --onnx-model-path '${ONNX_MODEL_PATH}' does not exist."
    exit 1
fi

mkdir -p "${OUTPUT_ENGINE_PATH}"

# -------------------- Build Engines --------------------

# VLLN-VLSelfAttention
echo "------------Building vlln_vl_self_attention Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=${ONNX_MODEL_PATH}/action_head/vlln_vl_self_attention.onnx \
  --saveEngine="${OUTPUT_ENGINE_PATH}/vlln_vl_self_attention.engine" \
  --minShapes=backbone_features:1x${MIN_LEN}x2048 \
  --optShapes=backbone_features:1x${OPT_LEN}x2048 \
  --maxShapes=backbone_features:8x${MAX_LEN}x2048 \
  > "${OUTPUT_ENGINE_PATH}/vlln_vl_self_attention.log" 2>&1

# DiT Model
echo "------------Building DiT Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=${ONNX_MODEL_PATH}/action_head/DiT.onnx \
  --saveEngine="${OUTPUT_ENGINE_PATH}/DiT.engine" \
  --minShapes=sa_embs:1x49x1536,vl_embs:1x${MIN_LEN}x2048,timesteps_tensor:1 \
  --optShapes=sa_embs:1x49x1536,vl_embs:1x${OPT_LEN}x2048,timesteps_tensor:1 \
  --maxShapes=sa_embs:8x49x1536,vl_embs:8x${MAX_LEN}x2048,timesteps_tensor:8 \
  > "${OUTPUT_ENGINE_PATH}/build_DiT.log" 2>&1

# State Encoder
echo "------------Building State Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=${ONNX_MODEL_PATH}/action_head/state_encoder.onnx \
  --saveEngine="${OUTPUT_ENGINE_PATH}/state_encoder.engine" \
  --minShapes=state:1x1x64,embodiment_id:1 \
  --optShapes=state:1x1x64,embodiment_id:1 \
  --maxShapes=state:8x1x64,embodiment_id:8 \
  > "${OUTPUT_ENGINE_PATH}/build_state_encoder.log" 2>&1

# Action Encoder
echo "------------Building Action Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=${ONNX_MODEL_PATH}/action_head/action_encoder.onnx \
  --saveEngine="${OUTPUT_ENGINE_PATH}/action_encoder.engine" \
  --minShapes=actions:1x16x${ACTION_DIM},timesteps_tensor:1,embodiment_id:1 \
  --optShapes=actions:1x16x${ACTION_DIM},timesteps_tensor:1,embodiment_id:1 \
  --maxShapes=actions:8x16x${ACTION_DIM},timesteps_tensor:8,embodiment_id:8 \
  > "${OUTPUT_ENGINE_PATH}/build_action_encoder.log" 2>&1

# Action Decoder
echo "------------Building Action Decoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=${ONNX_MODEL_PATH}/action_head/action_decoder.onnx \
  --saveEngine="${OUTPUT_ENGINE_PATH}/action_decoder.engine" \
  --minShapes=model_output:1x49x1024,embodiment_id:1 \
  --optShapes=model_output:1x49x1024,embodiment_id:1 \
  --maxShapes=model_output:8x49x1024,embodiment_id:8 \
  > "${OUTPUT_ENGINE_PATH}/build_action_decoder.log" 2>&1

# VLM-ViT
echo "------------Building VLM-ViT--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=${ONNX_MODEL_PATH}/eagle2/vit.onnx \
  --saveEngine="${OUTPUT_ENGINE_PATH}/vit.engine" \
  --minShapes=pixel_values:1x3x224x224,position_ids:1x256 \
  --optShapes=pixel_values:1x3x224x224,position_ids:1x256 \
  --maxShapes=pixel_values:8x3x224x224,position_ids:8x256 \
  > "${OUTPUT_ENGINE_PATH}/vit.log" 2>&1

# VLM-LLM
echo "------------Building VLM-LLM--------------------"
trtexec --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=${ONNX_MODEL_PATH}/eagle2/llm.onnx \
  --saveEngine="${OUTPUT_ENGINE_PATH}/llm.engine" \
  --minShapes=input_ids:1x${MIN_LEN},vit_embeds:1x256x1152,attention_mask:1x${MIN_LEN} \
  --optShapes=input_ids:1x${OPT_LEN},vit_embeds:1x${VIT_SEQ_LEN}x1152,attention_mask:1x${OPT_LEN} \
  --maxShapes=input_ids:8x${MAX_LEN},vit_embeds:8x${VIT_SEQ_LEN}x1152,attention_mask:8x${MAX_LEN} \
  > "${OUTPUT_ENGINE_PATH}/llm.log" 2>&1