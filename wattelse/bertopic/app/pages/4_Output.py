import streamlit as st
from pathlib import Path

from wattelse.bertopic.newsletter_features import generate_newsletter, md2html
from wattelse.bertopic.app.state_utils import (
    restore_widget_state,
    register_widget,
    save_widget_state,
)

restore_widget_state()

# Stop app if no topic is selected
if "topic_model" not in st.session_state.keys():
    st.error("Train a model to explore generated topics.", icon="🚨")
    st.stop()

if "newsletter_nb_topics" not in st.session_state:
    st.session_state["newsletter_nb_topics"] = 4
if "newsletter_nb_docs" not in st.session_state:
    st.session_state["newsletter_nb_docs"] = 3

# Newsletter params
col1, col2, col3 = st.columns(3)
register_widget("newsletter_nb_topics")
register_widget("newsletter_nb_docs")
register_widget("newsletter_improve_description")
with col1:
    st.slider(
        "Number of topics",
        min_value=1,
        max_value=10,
        key="newsletter_nb_topics",
        on_change=save_widget_state,
    )
with col2:
    st.slider(
        "Number of docs",
        min_value=1,
        max_value=6,
        key="newsletter_nb_docs",
        on_change=save_widget_state,
    )
with col3:
    st.toggle(
        "Improve topic description",
        value=False,
        key="newsletter_improve_description",
        on_change=save_widget_state,
    )

# Automatic newsletter
if st.session_state["split_by_paragraphs"]:
    df = st.session_state["initial_df"]
    df_split = st.session_state["timefiltered_df"]
else:
    df = st.session_state["timefiltered_df"]
    df_split = None
if st.button("Generate newsletter"):
    with st.spinner("Generating newsletter..."):
        md = generate_newsletter(
            topic_model=st.session_state["topic_model"],
            df=df,
            topics=st.session_state["topics"],
            df_split=df_split,
            top_n_topics=st.session_state["newsletter_nb_topics"],
            top_n_docs=st.session_state["newsletter_nb_docs"],
            improve_topic_description=st.session_state["newsletter_improve_description"],
        )
        # st.markdown(md)
        st.components.v1.html(
            md2html(md, Path(__file__).parent.parent.parent / "newsletter.css"),
            height=800,
            scrolling=True,
        )
