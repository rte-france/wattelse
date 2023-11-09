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
