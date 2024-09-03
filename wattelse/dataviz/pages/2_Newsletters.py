#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import streamlit as st

from wattelse.bertopic.app.data_utils import choose_data
from wattelse.bertopic.utils import OUTPUT_DIR

st.title("Browse generated newsletters")

# Load selected DataFrame
choose_data(OUTPUT_DIR, ["*.html"])

with open(st.session_state["data_folder"] / st.session_state["data_name"]) as f:
    html_content = f.read()

    st.components.v1.html(
        html_content,
        height=800,
        scrolling=True,
    )
