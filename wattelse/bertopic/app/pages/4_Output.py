import streamlit as st
from md2html import md2html

from wattelse.bertopic.ouput_features import generate_newsletter

# Stop app if no topic is selected
if "topic_model" not in st.session_state.keys():
    st.error("Train a model to explore generated topics.", icon="🚨")
    st.stop()

# Automatic newsletter
if st.button("Generate newsletter"):
    md = generate_newsletter(
        st.session_state["topic_model"],
        st.session_state["raw_df"],#load_data(st.session_state["data_name"]),
        st.session_state["topics"],
        df_split=st.session_state["timefiltered_df"],
    )
    #st.markdown(md)
    st.components.v1.html(md2html.render(md2html.parse_args(), md), height=800, scrolling=True)
