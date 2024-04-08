import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path

from wattelse.bertopic.app.state_utils import register_widget
from wattelse.bertopic.utils import (TEXT_COLUMN, TIMESTAMP_COLUMN, GROUPED_TIMESTAMP_COLUMN, URL_COLUMN, TITLE_COLUMN,
                                     CITATION_COUNT_COL, BASE_CACHE_PATH, load_data)
from wattelse.common.cache_utils import load_embeddings

DEFAULT_PARAMETERS = {
    "embedding_model_name": "antoinelouis/biencoder-camembert-base-mmarcoFR",
    "use_cached_embeddings": False,
    "bertopic_nr_topics": 0,
    "bertopic_top_n_words": 5,
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist": 0.0,
    "umap_metric": "cosine",
    "hdbscan_min_cluster_size": 10,
    "hdbscan_min_samples": 10,
    "hdbscan_metric": "euclidean",
    "hdbscan_cluster_selection_method": "eom",
    "hdbscan_cluster_selection_epsilon": 0.0, ######## NEW ########
    "hdbscan_max_cluster_size": 0, ######## NEW ########
    "hdbscan_allow_single_cluster": False, ######## NEW ########
    "countvectorizer_stop_words": "french",
    "countvectorizer_ngram_range": (1, 1),
    "ctfidf_reduce_frequent_words": True,
    "ctfidf_bm25_weighting":False
}

def initialize_default_parameters_keys():
    for k,v in DEFAULT_PARAMETERS.items():
        if k not in st.session_state:
            st.session_state[k] = v
        register_widget(k)


@st.cache_data
def load_data_wrapper(data_name: Path):
    return load_data(data_name)


def embedding_model_options():
    return {
        "embedding_model_name": st.selectbox("Name",
                                             ["dangvantuan/sentence-camembert-large", "paraphrase-multilingual-MiniLM-L12-v2",
                                              "sentence-transformers/all-mpnet-base-v2", "antoinelouis/biencoder-camembert-base-mmarcoFR"], key="embedding_model_name"),
        "use_cached_embeddings": st.toggle("Put embeddings in cache", key="use_cached_embeddings")
    }


def bertopic_options():
    return {
        "bertopic_nr_topics": st.number_input("nr_topics", min_value=0, key="bertopic_nr_topics"),
        "bertopic_top_n_words": st.number_input("top_n_words", min_value=1, key="bertopic_top_n_words"),
    }


def umap_options():
    return {
        "umap_n_neighbors": st.number_input("n_neighbors", min_value=1, key="umap_n_neighbors"),
        "umap_n_components": st.number_input("n_components", min_value=1, key="umap_n_components"),
        "umap_min_dist": st.number_input("min_dist", min_value=0.0, key="umap_min_dist", max_value=1.0),
        "umap_metric": st.selectbox("metric", ["cosine"], key="umap_metric"),
    }

def hdbscan_options():
    return {
        "hdbscan_min_cluster_size": st.number_input("min_cluster_size", min_value=1, key="hdbscan_min_cluster_size"),
        "hdbscan_min_samples": st.number_input("min_samples", min_value=1, key="hdbscan_min_samples"),
        "hdbscan_metric": st.selectbox("metric", ["euclidean"], key="hdbscan_metric"),
        "hdbscan_cluster_selection_method": st.selectbox("cluster_selectinumber_inputon_method", ["eom"], key="hdbscan_cluster_selection_method"),
        "hdbscan_cluster_selection_epsilon": st.number_input("cluster_selection_epsilon", min_value=0.0, key="hdbscan_cluster_selection_epsilon", format="%.2f", step=0.01), ######## NEW ########
        "hdbscan_max_cluster_size": st.number_input("max_cluster_size", min_value=0, key="hdbscan_max_cluster_size"), ######## NEW ########
        "hdbscan_allow_single_cluster": st.toggle("allow_single_cluster", key="hdbscan_allow_single_cluster") ######## NEW ########
    }


def countvectorizer_options():
    return {
        "countvectorizer_stop_words": st.selectbox("stop_words", ["french", "english", None], key="countvectorizer_stop_words"),
        "countvectorizer_ngram_range": st.selectbox(
            "ngram_range", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)], key="countvectorizer_ngram_range"
        ),
        "countvectorizer_min_df": st.number_input("min_df", min_value=1, value=2, key="countvectorizer_min_df"),
    }


def ctfidf_options():
    return {
        "ctfidf_reduce_frequent_words": st.toggle("reduce_frequent_words", key="ctfidf_reduce_frequent_words"),
        "ctfidf_bm25_weighting" : st.toggle("bm25_weighting", key="ctfidf_bm25_weighting"), ######## NEW ########
    }


@st.cache_data
def plot_topic_treemap(form_parameters, _topic_model, width=700):
    pass

@st.cache_data
def plot_2d_topics(form_parameters, _topic_model, width=700):
    return _topic_model.visualize_topics(width=width)


@st.cache_data
def plot_topics_hierarchy(form_parameters, _topic_model, width=700):
    return _topic_model.visualize_hierarchy(width=width)

def make_dynamic_topics_split(df, nr_bins):
    """Split docs into nr_bins and generate a common timestamp label into a new column"""
    # Ensure df is sorted by timestamp to get temporal split
    df = df.sort_values(TIMESTAMP_COLUMN, ascending=False)
    split_df = np.array_split(df, nr_bins)
    for split in split_df:
        split[GROUPED_TIMESTAMP_COLUMN] = split[TIMESTAMP_COLUMN].max()
    return pd.concat(split_df)


@st.cache_data
def compute_topics_over_time(form_parameters, _topic_model, df, nr_bins, new_df=None, new_nr_bins=None, new_topics=None):
    df = make_dynamic_topics_split(df, nr_bins)
    if new_nr_bins:
        new_df = make_dynamic_topics_split(new_df, new_nr_bins)
        df = pd.concat([df, new_df])
        _topic_model.topics_ += new_topics
    res = _topic_model.topics_over_time(
        df[TEXT_COLUMN],
        df[GROUPED_TIMESTAMP_COLUMN],
        global_tuning=False,
    )
    if new_nr_bins:
        _topic_model.topics_ = _topic_model.topics_[:-len(new_topics)]
    return res

def plot_topics_over_time(topics_over_time, dynamic_topics_list, topic_model, time_split=None, width=900):
    if dynamic_topics_list != "":
        if ":" in dynamic_topics_list:
            dynamic_topics_list = [i for i in range(int(dynamic_topics_list.split(":")[0]), int(dynamic_topics_list.split(":")[1]))]
        else:
            dynamic_topics_list = [int(i) for i in dynamic_topics_list.split(",")]
        fig = topic_model.visualize_topics_over_time(topics_over_time,
                                                     topics=dynamic_topics_list,
                                                     width=width,
                                                     title="",
                                                     )
        if time_split:
            fig.add_vline(x=time_split, line_width=3, line_dash="dash", line_color="black", opacity=1)
        return fig

def print_docs_for_specific_topic(df, topics, topic_number):
    # Select column available in DF
    columns_list = [col for col in [TITLE_COLUMN, TEXT_COLUMN, URL_COLUMN, TIMESTAMP_COLUMN, CITATION_COUNT_COL] if col in df.keys()]

    df = df.loc[pd.Series(topics)==topic_number][columns_list]
    for _, doc in df.iterrows():
        st.write(f"[{doc.title}]({doc.url})")

@st.cache_data
def transform_new_data(_topic_model, df, data_name, embedding_model_name, form_parameters=None, split_by_paragraphs=False):
    # Get DF embeddings
    if split_by_paragraphs:
        cache_path = BASE_CACHE_PATH / f"{embedding_model_name}_{data_name}_split_by_paragraphs.pkl"
    else:
        cache_path = BASE_CACHE_PATH / f"{embedding_model_name}_{data_name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = load_embeddings(cache_path)
    return _topic_model.transform(df[TEXT_COLUMN], embeddings=embeddings[df["index"]])
    
def plot_docs_reparition_over_time(df, freq):
    count = df.groupby(pd.Grouper(key="timestamp", freq=freq), as_index=False).size()
    count["timestamp"] = count["timestamp"].dt.strftime('%Y-%m-%d')

    fig = px.bar(count, x="timestamp", y="size")
    st.plotly_chart(fig, use_container_width=True)

def plot_remaining_docs_repartition_over_time(df_base, df_remaining, freq):
    # Concatenate df
    df = pd.concat([df_base, df_remaining])

    # Get split time value
    split_time = str(df_remaining["timestamp"].min())

    # Print aggregated docs
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    count = df.groupby(pd.Grouper(key="timestamp", freq=freq), as_index=False).size()
    count["timestamp"] = count["timestamp"].dt.strftime('%Y-%m-%d')
    # Split to set a different color to each DF
    count["category"] = ["Base" if time < split_time else "Remaining" for time in count["timestamp"]]

    fig = px.bar(
        count,
        x="timestamp",
        y="size",
        color="category",
        color_discrete_map={
        "Base" : "light blue", # default plotly color to match main page graphs
        "Remaining" : "orange",
    }
        )
    st.write(fig)
