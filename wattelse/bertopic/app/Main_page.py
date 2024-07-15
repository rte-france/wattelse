#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import streamlit as st
from loguru import logger
import pandas as pd
from wattelse.bertopic.app.data_utils import data_overview, choose_data
from wattelse.bertopic.topic_metrics import get_coherence_value, get_diversity_value
from wattelse.bertopic.app.train_utils import train_BERTopic_wrapper
import datetime
from bertopic import BERTopic
from typing import List
import plotly.graph_objects as go
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import re
import ast
import torch

from wattelse.bertopic.temporal_metrics_embedding import TempTopic

from wattelse.bertopic.utils import (
    TIMESTAMP_COLUMN,
    clean_dataset,
    split_df_by_paragraphs,
    split_df_by_paragraphs_v2,
    DATA_DIR,
    TEXT_COLUMN,
)

from wattelse.bertopic.app.state_utils import (
    register_widget,
    save_widget_state,
    restore_widget_state,
)

from wattelse.bertopic.app.app_utils import (
    embedding_model_options,
    bertopic_options,
    umap_options,
    hdbscan_options,
    countvectorizer_options,
    ctfidf_options,
    representation_model_options,
    plot_2d_topics,
    plot_topics_over_time,
    compute_topics_over_time,
    initialize_default_parameters_keys,
    load_data_wrapper,
)



def preprocess_text(text: str) -> str:
    """
    Preprocess French text by replacing hyphens and similar characters with spaces,
    removing specific prefixes, removing punctuations (excluding apostrophes, hyphens, and newlines),
    replacing special characters with a space (preserving accented characters, common Latin extensions, and newlines),
    normalizing superscripts and subscripts,
    splitting words containing capitals in the middle (while avoiding splitting fully capitalized words), 
    and replacing multiple spaces with a single space.
    
    Args:
        text (str): The input French text to preprocess.
    
    Returns:
        str: The preprocessed French text.
    """
    # Replace hyphens and similar characters with spaces
    text = re.sub(r'\b(-|/|;|:)', ' ', text)
    
    # Remove specific prefixes
    text = re.sub(r"\b(l'|L'|D'|d'|l’|L’|D’|d’)", ' ', text)
    
    # Remove punctuations (excluding apostrophes, hyphens, and newlines)
    # text = re.sub(r'[^\w\s\nàâçéèêëîïôûùüÿñæœ]', ' ', text)
    
    # Replace special characters with a space (preserving accented characters, common Latin extensions, and newlines)
    text = re.sub(r'[^\w\s\nàâçéèêëîïôûùüÿñæœ]', ' ', text)
    
    # Normalize superscripts and subscripts
    # Replace all superscripts and subscripts with their corresponding regular characters
    superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    text = text.translate(superscript_map)
    text = text.translate(subscript_map)

    # Split words that contain capitals in the middle but avoid splitting fully capitalized words
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text


def split_dataframe(split_option):
    if split_option == "Default split":
        st.session_state["split_df"] = split_df_by_paragraphs(st.session_state["raw_df"])
        st.session_state["split_by_paragraphs"] = True
    elif split_option == "Enhanced split":
        model_name = ast.literal_eval(st.session_state['parameters'])['embedding_model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_length = SentenceTransformer(model_name).get_max_seq_length()
        if max_length == 514: max_length = 512
        with st.spinner("Splitting the dataset..."):
            st.session_state["split_df"] = split_df_by_paragraphs_v2(
                dataset=st.session_state["raw_df"],
                tokenizer=tokenizer,
                max_length=max_length-2,
                min_length=0
            )
        st.session_state["split_by_paragraphs"] = True
        



def generate_model_name(base_name="topic_model"):
    """
    Generates a dynamic model name with the current date and time.
    If a base name is provided, it uses that instead of the default.
    """
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{base_name}_{current_datetime}"
    return model_name


def save_model_interface():
    st.write("## Save Model")

    # Optional text box for custom model name
    base_model_name = st.text_input("Enter a name for the model (optional):")

    # Button to save the model
    if st.button("Save Model"):
        if "topic_model" in st.session_state:
            dynamic_model_name = generate_model_name(base_model_name if base_model_name else "topic_model")
            model_save_path = f"./saved_models/{dynamic_model_name}"
            
            # Assuming the saving function and success/error messages are handled here
            try:
                st.session_state['topic_model'].save(model_save_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=True)
                st.success(f"Model saved successfully as {model_save_path}")
                st.session_state['model_saved'] = True
                st.balloons()
            except Exception as e:
                st.error(f"Failed to save the model: {e}")
        else:
            st.error("No model available to save. Please train a model first.")



################################################
################## MAIN PAGE ###################
################################################



# Wide layout
st.set_page_config(page_title="Wattelse® topic", layout="wide")




# Restore widget state
restore_widget_state()


### TITLE ###
st.title("Topic modelling")


# Initialize default parameters
initialize_default_parameters_keys()

if 'model_trained' not in st.session_state: st.session_state['model_trained'] = False
if 'model_saved' not in st.session_state: st.session_state['model_saved'] = False


# In the sidebar form
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
    
    with st.expander("Representation Models"):
        representation_model_options = representation_model_options()

    # Merge parameters in a dict and change type to str to make it hashable
    st.session_state["parameters"] = str(
        {
            **embedding_model_options,
            **bertopic_options,
            **umap_options,
            **hdbscan_options,
            **countvectorizer_options,
            **ctfidf_options,
            **representation_model_options,
        }
    )

    parameters_sidebar_clicked = st.form_submit_button(
        "Train model", type="primary", on_click=save_widget_state
    )

def select_data():
    st.write("## Data selection")

    choose_data(DATA_DIR, ["*.csv", "*.jsonl*"])
    
    ########## Adjusting to handle multiple files selection ##########
    if st.session_state["selected_files"]:
        loaded_dfs = []
        for file_path in st.session_state["selected_files"]:
            df = load_data_wrapper(file_path)
            df.sort_values(by=TIMESTAMP_COLUMN, ascending=False, inplace=True)
            loaded_dfs.append(df)

        st.session_state["raw_df"] = pd.concat(loaded_dfs) if len(loaded_dfs) > 1 else loaded_dfs[0]
    else:
        st.error("Please select at least one file to proceed.")
        st.stop()

    ########## Remove duplicates from raw_df ##########
    st.session_state["raw_df"] = st.session_state["raw_df"].drop_duplicates(subset=TEXT_COLUMN, keep='first')
    st.session_state["raw_df"].sort_values(by=[TIMESTAMP_COLUMN], ascending=True, inplace=True)
    st.session_state["initial_df"] = st.session_state["raw_df"].copy()
    

    st.divider()

    with st.container(border=True):
        # Select time range
        min_max = st.session_state["raw_df"][TIMESTAMP_COLUMN].agg(["min", "max"])
        register_widget("timestamp_range")
        if "timestamp_range" not in st.session_state:
            st.session_state["timestamp_range"] = (
                min_max["min"].to_pydatetime(),
                min_max["max"].to_pydatetime(),
            )

        timestamp_range = st.slider(
            "Select the range of timestamps you want to use for training",
            min_value=min_max["min"].to_pydatetime(),
            max_value=min_max["max"].to_pydatetime(),
            key="timestamp_range",
            on_change=save_widget_state,
        )

        # Filter text length parameter
        register_widget("min_text_length")
        st.number_input(
            "Select the minimum number of characters each document should contain",
            min_value=0,
            value=100,
            key="min_text_length",
            on_change=save_widget_state,
        )

        # Split DF by paragraphs parameter
        register_widget("split_option")
        split_options = ["No split", "Default split", "Enhanced split"]
        split_option = st.radio(
            "Select the split option",
            split_options,
            key="split_option",
            on_change=save_widget_state,
            help="""
            - No split: No splitting on the documents.
            - Default split: Split by paragraph (Warning: might produce paragraphs longer than embedding model's maximum supported input length).
            - Enhanced split: Split by paragraph. If a paragraph is longer than the embedding model's maximum input length, then split by sentence.
            """
        )
    

    if ("split_method" not in st.session_state or st.session_state["split_method"] != split_option or
        "prev_timestamp_range" not in st.session_state or st.session_state["prev_timestamp_range"] != timestamp_range or
        "prev_min_text_length" not in st.session_state or st.session_state["prev_min_text_length"] != st.session_state["min_text_length"]):
        st.session_state["split_method"] = split_option
        st.session_state["prev_timestamp_range"] = timestamp_range
        st.session_state["prev_min_text_length"] = st.session_state["min_text_length"]
        
        if split_option != "No split":
            split_dataframe(split_option)
        else: # If No Splitting is done
            st.session_state["split_df"] = st.session_state["raw_df"]
            st.session_state["split_by_paragraphs"] = False

    ########## Preprocess the text ##########
    st.session_state["split_df"][TEXT_COLUMN] = st.session_state["split_df"][TEXT_COLUMN].apply(preprocess_text)

    ########## Remove unwanted rows from split_df ##########
    st.session_state["split_df"] = st.session_state["split_df"][
        (st.session_state["split_df"][TEXT_COLUMN].str.strip() != "") &
        (st.session_state["split_df"][TEXT_COLUMN].apply(lambda x: len(re.findall(r'[a-zA-Z]', x)) >= 5))
    ]

    st.session_state["split_df"].reset_index(drop=True, inplace=True)
    st.session_state["split_df"]["index"] = st.session_state["split_df"].index

    # Filter dataset to select only text within time range
    st.session_state["timefiltered_df"] = st.session_state["split_df"].query(
        f"timestamp >= '{timestamp_range[0]}' and timestamp <= '{timestamp_range[1]}'"
    )

    # Clean dataset using min_text_length
    st.session_state["timefiltered_df"] = clean_dataset(
        st.session_state["timefiltered_df"],
        st.session_state["min_text_length"],
    )

    st.session_state["timefiltered_df"] = st.session_state["timefiltered_df"].reset_index(drop=True).reset_index()

    if st.session_state["timefiltered_df"].empty:
        st.error("Not enough remaining data after cleaning", icon="🚨")
        st.stop()
    else:
        st.info(f"Found {len(st.session_state['timefiltered_df'])} documents after final cleaning.")
        st.divider()

def train_model():
    if parameters_sidebar_clicked:
        if "timefiltered_df" in st.session_state and not st.session_state["timefiltered_df"].empty:
            
            full_dataset = st.session_state["timefiltered_df"]
            indices = full_dataset.index.tolist()

            (   st.session_state["topic_model"],
                st.session_state["topics"],
                _,
                st.session_state["embeddings"],
                st.session_state["token_embeddings"],
                st.session_state["token_strings"],
            ) = train_BERTopic_wrapper(
                dataset=full_dataset,
                indices=indices,
                form_parameters=st.session_state["parameters"],
                cache_base_name=st.session_state["data_name"]
                if st.session_state["split_method"] == "No split"
                else f'{st.session_state["data_name"]}_split_by_paragraphs',
            )
            
            st.info("Token embeddings aren't saved in cache and thus aren't loaded. Please make sure to train the model without using cached embeddings if you want correct and functional temporal visualizations.")
            
            temp = st.session_state["topic_model"].get_topic_info()
            st.session_state["topics_info"] = (
                temp[temp['Topic'] != -1]
            )  # exclude -1 topic from topic list

            # Optional : Not really useful here and takes time to calculate, uncomment if you want to calculate them
            # # Computes coherence value
            # coherence_score_type = "c_npmi"
            # coherence = get_coherence_value(
            #     st.session_state["topic_model"],
            #     st.session_state["topics"],
            #     st.session_state["timefiltered_df"][TEXT_COLUMN],
            #     coherence_score_type
            # )
            # diversity_score_type = "puw"
            # diversity = get_diversity_value(st.session_state["topic_model"],
            #                                 st.session_state["topics"],
            #                                 st.session_state["timefiltered_df"][TEXT_COLUMN],
            #                                 diversity_score_type="puw")
            
            # logger.info(f"Coherence score [{coherence_score_type}]: {coherence}")
            # logger.info(f"Diversity score [{diversity_score_type}]: {diversity}")
            
            st.session_state['model_trained'] = True
            if not st.session_state['model_saved']: st.warning('Don\'t forget to save your model!', icon="⚠️")
            
        else:
            st.error("No data available for training. Please ensure data is correctly loaded.")


# Load selected DataFrame
select_data()

# Data overview
data_overview(st.session_state["timefiltered_df"])

# Train model
train_model()

# Save the model button
save_model_interface()



