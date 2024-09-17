#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of Wattelse, a NLP application suite.
#

# Launches the VLLM 
# This script has to be run on a GPU server!


# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Specify the relative path to the config file
CONFIG_FILE="$SCRIPT_DIR/vllm_api.cfg"

# Use grep to extract config from the config file
HOST=$(grep -Po '(?<!#)host=\K.*' "$CONFIG_FILE")
PORT=$(grep -Po '(?<!#)port=\K.*' "$CONFIG_FILE")
MODEL_NAME=$(grep -Po '(?<!#)model_name=\K.*' "$CONFIG_FILE")
CUDA_VISIBLE_DEVICES=$(grep -Po '(?<!#)cuda_visible_devices=\K.*' "$CONFIG_FILE")


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m vllm.entrypoints.openai.api_server \
	--model $MODEL_NAME \
	--port $PORT \
	--host $HOST \
	--device auto \
	--worker-use-ray \
	--tensor-parallel-size 2 \
    --enforce-eager \
	--dtype=half # needed for T4 GPU that do not support Bfloat16

