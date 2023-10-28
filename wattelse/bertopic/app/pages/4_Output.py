import streamlit as st
from pathlib import Path

from wattelse.bertopic.ouput_features import generate_newsletter, md2html
from wattelse.bertopic.utils import load_data
from wattelse.bertopic.app.state_utils import restore_widget_state
from wattelse.bertopic.utils import DATA_DIR

restore_widget_state()

# Stop app if no topic is selected
if "topic_model" not in st.session_state.keys():
    st.error("Train a model to explore generated topics.", icon="🚨")
    st.stop()

# Automatic newsletter
if st.button("Generate newsletter"):
    md = generate_newsletter(
        st.session_state["topic_model"],
        load_data(f"{DATA_DIR}/{st.session_state['data_name']}"),
        st.session_state["topics"],
        df_split=st.session_state["timefiltered_df"],
    )
    # st.markdown(md)
    st.components.v1.html(
        md2html(md, Path(__file__).parent.parent.parent / "newsletter.css"),
        height=800,
        scrolling=True,
    )
