#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of Wattelse, a NLP application suite.
#

# Checks the PIDs of processes serving the LLM model, if not empty the service is running
sudo pgrep  -f 'ollama'