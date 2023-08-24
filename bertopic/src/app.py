import ast
import os

import pandas as pd
import streamlit as st
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from metrics import TopicMetrics
from utils import TEXT_COLUMN, TIMESTAMP_COLUMN, DATA_DIR, clean_dataset
from app_utils import load_data
from app_utils import (
    data_cleaning_options,
    embedding_model_options,
    bertopic_options,
    umap_options,
    hdbscan_options,
    countvectorizer_options,
    ctfidf_options,
    plot_barchart,
    plot_topics_hierarchy,
    plot_topics_over_time,
    print_topics,
    print_search_results,
    print_new_document_probs,
    compute_topics_over_time,
    print_docs_for_specific_topic,
)

from train import train_BERTopic, EmbeddingModel


@st.cache_data
def BERTopic_train(dataset: pd.DataFrame, form_parameters):

    # Transform form_parameters from str to dict (dict is not yet hashable using Streamlit)
    form_parameters = ast.literal_eval(form_parameters)

    # Step 1 - Embedding model
    embedding_model = EmbeddingModel(form_parameters["embedding_model_name"])

    # Step 2 - Dimensionality reduction algorithm
    umap_model = UMAP(
        n_neighbors=form_parameters["umap_n_neighbors"],
        n_components=form_parameters["umap_n_components"],
        min_dist=form_parameters["umap_min_dist"],
        metric=form_parameters["umap_metric"],
    )

    # Step 3 - Clustering algorithm
    hdbscan_model = HDBSCAN(
        min_cluster_size=form_parameters["hdbscan_min_cluster_size"],
        min_samples=form_parameters["hdbscan_min_samples"],
        metric=form_parameters["hdbscan_metric"],
        cluster_selection_method=form_parameters["hdbscan_cluster_selection_method"],
        prediction_data=True,
    )

    # Step 4 - Count vectorizer
    stop_words = (
        stopwords.words(form_parameters["countvectorizer_stop_words"])
        if form_parameters["countvectorizer_stop_words"]
        else None
    )

    vectorizer_model = CountVectorizer(
        stop_words=stop_words, ngram_range=form_parameters["countvectorizer_ngram_range"],
    )

    # Step 5 - c-TF-IDF model
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=form_parameters["ctfidf_reduce_frequent_words"]
    )

    return train_BERTopic(
        dataset[TEXT_COLUMN],
        embedding_model,
        umap_model,
        hdbscan_model,
        vectorizer_model,
        ctfidf_model,
        top_n_words=form_parameters["bertopic_top_n_words"],
        nr_topics=form_parameters["bertopic_nr_topics"] if form_parameters["bertopic_nr_topics"] > 0 else None,
        use_cache=form_parameters["use_cached_embeddings"],
    )


### DEFINE TITLE ###

st.title("RTE - BERTopic")

st.write("## Data selection")


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

# Select box with every file saved in "./data/" as options.
# As long as no data is selected, the app breaks.
data_options = ["None"] + os.listdir(DATA_DIR)
data_name = st.selectbox("Select data to continue", data_options)

if data_name == "None":
    st.stop()

df = load_data(data_name)
st.write(f"Found {len(df)} documents.")
st.write(df.head())


### TRAIN MODEL

if (
    not parameters_sidebar_clicked
    and not ("search_term" in st.session_state)
    and not ("new_document" in st.session_state)
):
    st.stop()

# Clean dataset
df = clean_dataset(df, ast.literal_eval(st.session_state.parameters)["min_text_length"])

# Train
topics, probs, topic_model = BERTopic_train(df, st.session_state.parameters)

most_likely_topic_per_doc = probs.argmax(axis=1) # select most likely topic per document, match outliers (topic -1) documents to actual topic

### DISPLAY RESULTS

st.write("## Results")

# Barchart
st.write(plot_barchart(st.session_state.parameters, topic_model))

# HDBSCAN Dendrogram
st.write(plot_topics_hierarchy(st.session_state.parameters, topic_model))

# Dynamic topic modelling
if TIMESTAMP_COLUMN in df.keys():
    st.write("## Dynamic topic modelling")

    #Parameters
    st.text_input("Topics list (format 1,12,52 or 1:20)", key="dynamic_topics_list", value="0:10")
    st.number_input("nr_bins", min_value=1, value=20, key="nr_bins")

    # Compute topics over time
    st.session_state["topics_over_time"] = compute_topics_over_time(st.session_state.parameters, topic_model, df, nr_bins=st.session_state.nr_bins)

    # Visualize
    st.write(plot_topics_over_time(st.session_state.topics_over_time, st.session_state.dynamic_topics_list, topic_model))


### USE THE TRAINED MODEL

# Explore topics

st.write("## Topics information")
print_topics(topic_model)


# Find docs belonging to a specific topic

st.write("## Find docs belonging to a specific topic")
st.number_input("Topic number", min_value=0, value=0, key="topic_number")
print_docs_for_specific_topic(df, most_likely_topic_per_doc, st.session_state.topic_number)


# Find most similar topics

st.write("## Find most similar topic using search term")
st.text_input("Enter search terms:", key="search_term")

if "search_term" in st.session_state:
    print_search_results(st.session_state.search_term, topic_model)


# Input

st.write("## Make topic prediction for a new document:")
st.text_area("Copy/Paste your document:", key="new_document")

if "new_document" in st.session_state:
    print_new_document_probs(st.session_state.new_document, topic_model)


# Topic maps
tw=0.02
st.write("## Topic map")
topic_metrics = TopicMetrics(topic_model, st.session_state.topics_over_time)
st.plotly_chart(topic_metrics.plot_TIM_map(tw))