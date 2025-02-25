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
CONFIG_FILE="$SCRIPT_DIR/rag_orchestrator.cfg"

# Use grep to extract config from the config file
PORT=$(grep -Po '(?<!#)port=\K.*' "$CONFIG_FILE")

uvicorn wattelse.api.rag_orchestrator.main:app --port=$PORT --reload --host 0.0.0.0