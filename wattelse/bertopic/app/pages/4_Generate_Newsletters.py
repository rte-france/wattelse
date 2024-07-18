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

restore_widget_state()

SUMMARIZER_OPTIONS_MAPPER = {
    "AbstractiveSummarizer": AbstractiveSummarizer,
    "GPTSummarizer": GPTSummarizer,
    "FastchatLLMSummarizer": FastchatLLMSummarizer,
    "ExtractiveSummarizer": ExtractiveSummarizer,
}

EXPORT_BASE_FOLDER = Path(__file__).parent.parent / 'exported_topics'

def generate_newsletter_wrapper():
    summarizer_class = None if st.session_state["summary_mode"] == "none" else SUMMARIZER_OPTIONS_MAPPER[st.session_state["summarizer_classname"]]
    
    md_content, html_content, date_min, date_max, html_file_path, json_file_path = generate_newsletter(
        topic_model=st.session_state["topic_model"],
        df=df,
        topics=st.session_state["topics"],
        df_split=df_split,
        top_n_topics=None if st.session_state["all_topics"] else st.session_state["newsletter_nb_topics"],
        top_n_docs=None if st.session_state["all_docs"] else st.session_state["newsletter_nb_docs"],
        improve_topic_description=st.session_state["newsletter_improve_description"],
        summarizer_class=summarizer_class,
        summary_mode=st.session_state['summary_mode'],
        export_base_folder=EXPORT_BASE_FOLDER,
        batch_size=10
    )
    return md_content, html_content, date_min, date_max, html_file_path, json_file_path

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
if "summary_mode" not in st.session_state:
    st.session_state["summary_mode"] = 'none'
if "all_topics" not in st.session_state:
    st.session_state["all_topics"] = False
if "all_docs" not in st.session_state:
    st.session_state["all_docs"] = False

# Newsletter params
with st.sidebar.form("newsletter_parameters"):
    register_widget("all_topics")
    register_widget("all_docs")
    register_widget("newsletter_nb_topics")
    register_widget("newsletter_nb_docs")
    register_widget("newsletter_improve_description")
    register_widget("summarizer_classname")
    register_widget("summary_mode")
    
    st.checkbox("Include all topics", key="all_topics")
    st.checkbox("Include all documents per topic", key="all_docs")
    
    st.slider(
        "Number of topics",
        min_value=1,
        max_value=10,
        key="newsletter_nb_topics",
    )
    
    st.slider(
        "Number of docs per topic",
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
        "Summary mode",
        ['none', 'topic', 'document'],
        key="summary_mode",
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
        md_content, html_content, date_min, date_max, html_file_path, json_file_path = generate_newsletter_wrapper()
    
    if md_content and html_content:
        st.success(f"Newsletter generated successfully! You can find the HTML version at: {html_file_path}")
        st.success(f"JSON data saved at: {json_file_path}")
        
        st.markdown("## Newsletter Preview (HTML version)")
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.error("Failed to generate the newsletter. Please check the logs for more information.")