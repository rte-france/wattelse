import ast
import os

import streamlit as st

from utils import TIMESTAMP_COLUMN, DATA_DIR, clean_dataset
from app_utils import load_data
from app_utils import (
    data_cleaning_options,
    embedding_model_options,
    bertopic_options,
    umap_options,
    hdbscan_options,
    countvectorizer_options,
    ctfidf_options,
    plot_2d_topics,
    plot_topics_over_time,
    compute_topics_over_time,
)

from train_utils import train_BERTopic_wrapper

### TITLE ###

st.title("BERTopic")


### SIDEBAR OPTIONS ###

with st.sidebar.form("parameters_sidebar"):

    st.title("Parameters")

    with st.expander("Data cleaning options"):
        data_cleaning_options = data_cleaning_options()

    with st.expander("Embedding model"):
        embedding_model_options = embedding_model_options()

    with st.expander("Topics"):
        bertopic_options = bertopic_options()

    with st.expander("UMAP"):
        umap_options = umap_options()

    with st.expander("HDBSCAN"):
        hdbscan_options = hdbscan_options()

    with st.expander("Count Vectorizer"):
        countvectorizer_options = countvectorizer_options()

    with st.expander("c-TF-IDF"):
        ctfidf_options = ctfidf_options()

    # Merge parameters in a dict and change type to str to make it hashable
    st.session_state["parameters"] = str(
        {
            **data_cleaning_options,
            **embedding_model_options,
            **bertopic_options,
            **umap_options,
            **hdbscan_options,
            **countvectorizer_options,
            **ctfidf_options,
        }
    )

    parameters_sidebar_clicked = st.form_submit_button("Train model")


### DATA SELECTION AND LOADING ###

st.write("## Data selection")

# Select box with every file saved in DATA_DIR as options
data_options = ["None"] + os.listdir(DATA_DIR)
st.session_state["data_name"] = st.selectbox("Select data to continue:", data_options, index=st.session_state.get("data_name_index", 0))
st.session_state["data_name_index"] = data_options.index(st.session_state["data_name"])

# Stop the app as long as no data is selected
if st.session_state["data_name"] == "None":
    st.stop()

# Load selected DataFrame
st.session_state["raw_df"] = load_data(st.session_state["data_name"]).sort_values(by=TIMESTAMP_COLUMN, ascending=False)
st.write(f"Found {len(st.session_state['raw_df'])} documents.")
st.write(st.session_state["raw_df"].head())


### TRAIN MODEL ###

if parameters_sidebar_clicked:

    # Clean dataset
    st.session_state["df"] = clean_dataset(st.session_state["raw_df"], ast.literal_eval(st.session_state["parameters"])["min_text_length"])

    # Train
    _, probs, st.session_state["topic_model"] = train_BERTopic_wrapper(st.session_state["df"], st.session_state["parameters"])
    st.session_state["topic_per_doc"] = probs.argmax(axis=1) # select most likely topic per document to match outliers (topic -1) documents to actual topic
    st.session_state["topics_list"] = st.session_state["topic_model"].get_topic_info().iloc[1:] # exclude -1 topic from topic list

### PRINT GLOBAL RESULTS ###

if not ("topic_model" in st.session_state.keys()):
    st.stop()

# 2d plot
st.write("## Overall results")
st.write(plot_2d_topics(st.session_state.parameters, st.session_state["topic_model"]))

# Dynamic topic modelling
if TIMESTAMP_COLUMN in st.session_state["df"].keys():
    st.write("## Dynamic topic modelling")

    #Parameters
    st.text_input("Topics list (format 1,12,52 or 1:20)", key="dynamic_topics_list", value="0:10")
    st.number_input("nr_bins", min_value=1, value=20, key="nr_bins")

    # Compute topics over time only when train button is clicked
    if parameters_sidebar_clicked:
        st.session_state["topics_over_time"] = compute_topics_over_time(st.session_state["parameters"], st.session_state["topic_model"], st.session_state["df"], nr_bins=st.session_state["nr_bins"])

    # Visualize
    st.write(plot_topics_over_time(st.session_state["topics_over_time"], st.session_state["dynamic_topics_list"], st.session_state["topic_model"]))