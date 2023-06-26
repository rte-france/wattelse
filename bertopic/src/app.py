import ast
import os
import streamlit as st
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from utils import TEXT_COLUMN, TIMESTAMP_COLUMN, DATA_DIR
from app_utils import load_data
from app_utils import (
    embedding_model_options,
    bertopic_options,
    umap_options,
    hdbscan_options,
    countvectorizer_options,
    ctfidf_options,
    plot_barchart,
    plot_topics_hierarchy,
    plot_topics_over_time,
    print_search_results,
    print_new_document_probs,
)

from train import train_BERTopic, EmbeddingModel


@st.cache_data
def BERTopic_train(docs, form_parameters):

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
        docs,
        embedding_model,
        umap_model,
        hdbscan_model,
        vectorizer_model,
        ctfidf_model,
        top_n_words=form_parameters["bertopic_top_n_words"],
        nr_topics=form_parameters["bertopic_nr_topics"],
    )


### DEFINE TITLE ###

st.title("RTE - BERTopic")

st.write("## Data selection")


### SIDEBAR OPTIONS ###

with st.sidebar.form("parameters_sidebar"):

    st.title("Parameters")

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

topics, probs, topic_model = BERTopic_train(df[TEXT_COLUMN], st.session_state.parameters)


### DISPLAY RESULTS

st.write("## Results")

st.write(plot_barchart(st.session_state.parameters, topic_model))

st.write(plot_topics_hierarchy(st.session_state.parameters, topic_model))

if TIMESTAMP_COLUMN in df.keys():
    st.write(plot_topics_over_time(st.session_state.parameters, topic_model, df))


### USE THE TRAINED MODEL

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
