#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of Wattelse, a NLP application suite.
#

# Kills existing embedding api

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Specify the relative path to the config file
CONFIG_FILE="$SCRIPT_DIR/embedding_api.cfg"

# Find listening port
PORT=$(grep -Po '(?<!#)port=\K.*' "$CONFIG_FILE")

# Uses sudo to kill processes possibly launched by another user
kill -9 $(lsof -t -i :$PORT)
