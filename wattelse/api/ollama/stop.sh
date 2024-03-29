#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of Wattelse, a NLP application suite.
#

# Kills existing LLM service
# Uses sudo to kill processes possibly launched by another user
sudo pkill -SIGINT -c  -f 'ollama'
