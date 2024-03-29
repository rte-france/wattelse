#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of Wattelse, a NLP application suite.
#

# Start embedding API

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Specify the relative path to the config file
CONFIG_FILE="$SCRIPT_DIR/embedding_api.cfg"

# Use grep to extract config from the config file
PORT=$(grep -Po '(?<!#)port=\K.*' "$CONFIG_FILE")
CUDA_VISIBLE_DEVICES=$(grep -Po '(?<!#)cuda_visible_devices=\K.*' "$CONFIG_FILE")

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uvicorn wattelse.api.embedding.fastapi_embedding:app --port=$PORT --reload
