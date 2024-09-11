#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

OPENAI_API_KEY=$OPENAI_API_KEY_VEILLE CUDA_VISIBLE_DEVICES=0 streamlit run  --theme.primaryColor royalblue --server.port 8686 veille_analyse.py