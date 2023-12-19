import streamlit as st
from pathlib import Path

from wattelse.bertopic.newsletter_features import generate_newsletter, md2html
from wattelse.bertopic.app.state_utils import (
    restore_widget_state,
    register_widget,
    save_widget_state,
)
from wattelse.summary import (
    GPTSummarizer,
    AbstractiveSummarizer,
    ExtractiveSummarizer,
    LocalLLMSummarizer,
)

restore_widget_state()

SUMMARIZER_OPTIONS_MAPPER = {
    "AbstractiveSummarizer": AbstractiveSummarizer,
    "GPTSummarizer": GPTSummarizer,
    "LocalLLMSummarizer": LocalLLMSummarizer,
    "ExtractiveSummarizer": ExtractiveSummarizer,
}


def generate_newsletter_wrapper():
    return generate_newsletter(
        topic_model=st.session_state["topic_model"],
        df=df,
        topics=st.session_state["topics"],
        df_split=df_split,
        top_n_topics=st.session_state["newsletter_nb_topics"],
        top_n_docs=st.session_state["newsletter_nb_docs"],
        improve_topic_description=st.session_state["newsletter_improve_description"],
        summarizer_class=SUMMARIZER_OPTIONS_MAPPER[
            st.session_state["summarizer_classname"]
        ],
    )


# Stop app if no topic is selected
if "topic_model" not in st.session_state.keys():
    st.error("Train a model to explore generated topics.", icon="🚨")
    st.stop()

# Title
st.title("Automatic newsletter generation")

if "newsletter_nb_topics" not in st.session_state:
    st.session_state["newsletter_nb_topics"] = 4
if "newsletter_nb_docs" not in st.session_state:
    st.session_state["newsletter_nb_docs"] = 3
if "summarizer_classname" not in st.session_state:
    st.session_state["summarizer_classname"] = list(SUMMARIZER_OPTIONS_MAPPER.keys())[0]

# Newsletter params
with st.sidebar.form("newsletter_parameters"):
    register_widget("newsletter_nb_topics")
    register_widget("newsletter_nb_docs")
    register_widget("newsletter_improve_description")
    register_widget("summarizer_classname")
    st.slider(
        "Number of topics",
        min_value=1,
        max_value=10,
        key="newsletter_nb_topics",
    )

    st.slider(
        "Number of docs",
        min_value=1,
        max_value=6,
        key="newsletter_nb_docs",
    )

    st.toggle(
        "Improve topic description",
        value=False,
        key="newsletter_improve_description",
    )

    st.selectbox(
        "Summarizer class",
        list(SUMMARIZER_OPTIONS_MAPPER.keys()),
        key="summarizer_classname",
    )

    newsletter_parameters_clicked = st.form_submit_button(
        "Generate newsletter", type="primary", on_click=save_widget_state
    )

if newsletter_parameters_clicked:
    # Automatic newsletter
    if st.session_state["split_by_paragraphs"]:
        df = st.session_state["initial_df"]
        df_split = st.session_state["timefiltered_df"]
    else:
        df = st.session_state["timefiltered_df"]
        df_split = None
    with st.spinner("Generating newsletter..."):
        st.session_state["newsletter"] = generate_newsletter_wrapper()

if "newsletter" in st.session_state:
    st.components.v1.html(
        md2html(
            st.session_state["newsletter"][0],
            Path(__file__).parent.parent.parent / "newsletter.css",
        ),
        height=800,
        scrolling=True,
    )
