#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of Wattelse, a NLP application suite.
#

# Launches the LLM (FastChat) service
# This script has to be run on a GPU server!
# use the option "--load-8bit" to launch it in 8-bit mode


# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Specify the relative path to the config file
CONFIG_FILE="$SCRIPT_DIR/fastchat_api.cfg"

# Use grep to extract config from the config file
HOST=$(grep -Po '(?<!#)host=\K.*' "$CONFIG_FILE")
PORT=$(grep -Po '(?<!#)port=\K.*' "$CONFIG_FILE")
PORT_CONTROLLER=$(grep -Po '(?<!#)port_controller=\K.*' "$CONFIG_FILE")
PORT_WORKER=$(grep -Po '(?<!#)port_worker=\K.*' "$CONFIG_FILE")
MODEL_NAME=$(grep -Po '(?<!#)model_name=\K.*' "$CONFIG_FILE")
CUDA_VISIBLE_DEVICES=$(grep -Po '(?<!#)cuda_visible_devices=\K.*' "$CONFIG_FILE")


# launch controller
python3 -m fastchat.serve.controller --host $HOST --port $PORT_CONTROLLER &

# launch worker
if [ -z "$1" ]; then
  # no args provided
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m fastchat.serve.model_worker --host $HOST --port $PORT_WORKER --worker-address http://$HOST:$PORT_WORKER --controller-address http://$HOST:$PORT_CONTROLLER --model-path $MODEL_NAME --model-names $MODEL_NAME --num-gpus 2 --max-gpu-memory 8GiB&
else
  # we assume any args means '--load-8-bit' (NB. it seems that if we do not change the num-gpus values, the parameter is not taken into account
  echo "Starting service in 8-bit mode"
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m fastchat.serve.model_worker --host $HOST --port $PORT_WORKER --worker-address http://$HOST:$PORT_WORKER --controller-address http://$HOST:$PORT_CONTROLLER --model-path $MODEL_NAME --model-names $MODEL_NAME --num-gpus 1 --load-8bit&
fi

# launch API server
python3 -m fastchat.serve.openai_api_server --host $HOST --port $PORT --controller-address http://$HOST:$PORT_CONTROLLER &
