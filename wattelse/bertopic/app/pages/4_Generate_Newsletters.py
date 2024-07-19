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
    FastchatLLMSummarizer,
)

# Restore widget state
restore_widget_state()

# Define summarizer options
SUMMARIZER_OPTIONS_MAPPER = {
    "AbstractiveSummarizer": AbstractiveSummarizer,
    "GPTSummarizer": GPTSummarizer,
    "FastchatLLMSummarizer": FastchatLLMSummarizer,
    "ExtractiveSummarizer": ExtractiveSummarizer,
}

def generate_newsletter_wrapper():
    """Wrapper function to generate newsletter based on user settings."""
    top_n_topics = None if st.session_state["newsletter_all_topics"] else st.session_state["newsletter_nb_topics"]
    top_n_docs = None if st.session_state["newsletter_all_docs"] else st.session_state["newsletter_nb_docs"]
    
    return generate_newsletter(
        topic_model=st.session_state["topic_model"],
        df=df,
        topics=st.session_state["topics"],
        df_split=df_split,
        top_n_topics=top_n_topics,
        top_n_docs=top_n_docs,
        improve_topic_description=st.session_state["newsletter_improve_description"],
        summarizer_class=SUMMARIZER_OPTIONS_MAPPER[st.session_state["summarizer_classname"]],
        summary_mode=st.session_state['summary_mode'],
    )

# Check if a topic model exists
if "topic_model" not in st.session_state:
    st.error("Train a model to explore generated topics.", icon="🚨")
    st.stop()

# Title
st.title("Automatic newsletter generation")

# Initialize session state variables
default_values = {
    "newsletter_nb_topics": 4,
    "newsletter_nb_docs": 3,
    "summarizer_classname": list(SUMMARIZER_OPTIONS_MAPPER.keys())[0],
    "summary_mode": 'topic',
    "newsletter_all_topics": False,
    "newsletter_all_docs": False,
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Newsletter parameters sidebar
with st.sidebar.form("newsletter_parameters"):
    register_widget("newsletter_all_topics")
    register_widget("newsletter_all_docs")
    register_widget("newsletter_nb_topics")
    register_widget("newsletter_nb_docs")
    register_widget("newsletter_improve_description")
    register_widget("summarizer_classname")
    register_widget("summary_mode")

    st.checkbox("Include all topics", key="newsletter_all_topics")
    st.slider("Number of topics", min_value=1, max_value=20, key="newsletter_nb_topics")

    st.checkbox("Include all documents per topic", key="newsletter_all_docs")
    st.slider("Number of docs per topic", min_value=1, max_value=10, key="newsletter_nb_docs")

    st.toggle("Improve topic description", value=False, key="newsletter_improve_description")
    st.selectbox("Summary mode", ['topic', 'document', 'none'], key="summary_mode")
    st.selectbox("Summarizer class", list(SUMMARIZER_OPTIONS_MAPPER.keys()), key="summarizer_classname")
    
    newsletter_parameters_clicked = st.form_submit_button(
        "Generate newsletter", type="primary", on_click=save_widget_state
    )

# Generate newsletter when button is clicked
if newsletter_parameters_clicked:
    if st.session_state["split_by_paragraphs"]:
        df = st.session_state["initial_df"]
        df_split = st.session_state["timefiltered_df"]
    else:
        df = st.session_state["timefiltered_df"]
        df_split = None
    
    with st.spinner("Generating newsletter..."):
        st.session_state["newsletter"] = generate_newsletter_wrapper()

# Display generated newsletter
if "newsletter" in st.session_state:
    st.components.v1.html(
        md2html(
            st.session_state["newsletter"][0],
            Path(__file__).parent.parent.parent / "newsletter.css",
        ),
        height=800,
        scrolling=True,
    )
    
# TODO: Properly handle the streamlit interface's session state to automatically gray out the number of topics/document sliders if all topics/all documents checkboxes are activated.
# TODO: Currently, the gpt key being used in is a .cfg, it's preferable to modify the code in order to use directly from the environment's variables.