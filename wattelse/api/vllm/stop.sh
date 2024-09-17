#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of Wattelse, a NLP application suite.
#

# Command used to launch the service.
SERVICE_COMMAND="python -m vllm.entrypoints.openai.api_server"

# Find and kill the process
if pgrep -f "$SERVICE_COMMAND" > /dev/null
then
    pkill -f "$SERVICE_COMMAND"
fi