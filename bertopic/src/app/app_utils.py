import pandas as pd
import plotly.express as px
import streamlit as st
from utils import TEXT_COLUMN, TIMESTAMP_COLUMN, URL_COLUMN, TITLE_COLUMN, DATA_DIR, file_to_pd
from state_utils import register_widget

DEFAULT_PARAMETERS = {
    "min_text_length": 300,
    "embedding_model_name": "dangvantuan/sentence-camembert-large",
    "use_cached_embeddings": True,
    "bertopic_nr_topics": 0,
    "bertopic_top_n_words": 10,
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist": 0.0,
    "umap_metric": "cosine",
    "hdbscan_min_cluster_size": 10,
    "hdbscan_min_samples": 10,
    "hdbscan_metric": "euclidean",
    "hdbscan_cluster_selection_method": "eom",
    "countvectorizer_stop_words": "french",
    "countvectorizer_ngram_range": (1, 1),
    "ctfidf_reduce_frequent_words": True,
}
def initialize_default_parameters_keys():
    for k,v in DEFAULT_PARAMETERS.items():
        if k not in st.session_state:
            st.session_state[k] = v
        register_widget(k)


@st.cache_data
def load_data(data_name: str):
    df = file_to_pd(data_name, base_dir=DATA_DIR)
    # convert timestamp column
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    return df



def data_cleaning_options():
    return {
        "min_text_length": st.number_input("min_text_length (#chars)", min_value=0, key="min_text_length"),
    }

def embedding_model_options():
    return {
        "embedding_model_name": st.selectbox("Name",
                                             ["dangvantuan/sentence-camembert-large", "paraphrase-multilingual-MiniLM-L12-v2",
                                              "sentence-transformers/all-mpnet-base-v2"], key="embedding_model_name"),
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
        "hdbscan_cluster_selection_method": st.selectbox("cluster_selection_method", ["eom"], key="hdbscan_cluster_selection_method"),
    }


def countvectorizer_options():
    return {
        "countvectorizer_stop_words": st.selectbox("stop_words", ["french", "english", None], key="countvectorizer_stop_words"),
        "countvectorizer_ngram_range": st.selectbox(
            "ngram_range", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)], key="countvectorizer_ngram_range"
        ),
    }


def ctfidf_options():
    return {
        "ctfidf_reduce_frequent_words": st.toggle("reduce_frequent_words", key="ctfidf_reduce_frequent_words")
    }


@st.cache_data
def plot_2d_topics(form_parameters, _topic_model, width=700):
    return _topic_model.visualize_topics(width=width)


@st.cache_data
def plot_topics_hierarchy(form_parameters, _topic_model, width=700):
    return _topic_model.visualize_hierarchy(width=width)


@st.cache_data
def compute_topics_over_time(form_parameters, _topic_model, df, nr_bins=50):
    return _topic_model.topics_over_time(
        df[TEXT_COLUMN], df[TIMESTAMP_COLUMN], nr_bins=nr_bins, global_tuning=False
    )

def plot_topics_over_time(topics_over_time, dynamic_topics_list, topic_model, width=700):
    if dynamic_topics_list != "":
        if ":" in dynamic_topics_list:
            dynamic_topics_list = [i for i in range(int(dynamic_topics_list.split(":")[0]), int(dynamic_topics_list.split(":")[1]))]
        else:
            dynamic_topics_list = [int(i) for i in dynamic_topics_list.split(",")]
        return topic_model.visualize_topics_over_time(topics_over_time, topics=dynamic_topics_list, width=width)

def print_docs_for_specific_topic(df, most_likely_topic_per_doc, topic_number):
    # Select column available in DF
    columns_list = [col for col in [TITLE_COLUMN, TEXT_COLUMN, URL_COLUMN, TIMESTAMP_COLUMN, CITATION_COUNT_COL] if col in df.keys()]
    
    st.dataframe(
        df.loc[most_likely_topic_per_doc==topic_number][columns_list],
        column_config={"url": st.column_config.LinkColumn()}
        )
    
def plot_docs_reparition_over_time(df, freq):
    df.loc[0, "timestamp"] = df["timestamp"].iloc[0].normalize()

    count = df.groupby(pd.Grouper(key="timestamp", freq=freq), as_index=False).size()
    count["timestamp"] = count["timestamp"].dt.strftime('%Y-%m-%d')

    fig = px.bar(count, x="timestamp", y="size")
    st.write(fig)

def plot_remaining_docs_repartition_over_time(df_base, df_remaining, freq):
    # Concatenate df
    df = pd.concat([df_base, df_remaining])

    # Get split time value
    split_time = str(df_remaining["timestamp"].min())

    # Print aggregated docs
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.loc[0, "timestamp"] = df["timestamp"].iloc[0].normalize()

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

