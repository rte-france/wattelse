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

from wattelse.bertopic.utils import (
    TIMESTAMP_COLUMN,
    clean_dataset,
    split_df_by_paragraphs,
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
    plot_2d_topics,
    plot_topics_over_time,
    compute_topics_over_time,
    initialize_default_parameters_keys,
    load_data_wrapper,
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

        # Concatenate all loaded DataFrames if there's more than one, else just use the single loaded DataFrame
        st.session_state["raw_df"] = pd.concat(loaded_dfs).reset_index(drop=True).reset_index() if len(loaded_dfs) > 1 else loaded_dfs[0].reset_index(drop=True).reset_index()
    else:
        st.error("Please select at least one file to proceed.")
        st.stop()

    ########## Remaining parts of the function do not need adjustments for multi-file logic ##########
    
    # Filter text length parameter
    register_widget("min_text_length")
    st.number_input(
        "Select the minimum number of characters each docs should contain",
        min_value=0,
        key="min_text_length",
        on_change=save_widget_state,
    )

    # Split DF by paragraphs parameter
    register_widget("split_by_paragraphs")
    st.toggle(
        "Split texts by paragraphs",
        value=False,
        key="split_by_paragraphs",
        on_change=save_widget_state,
    )

    # Split if parameter is selected
    if st.session_state["split_by_paragraphs"]:
        st.session_state["initial_df"] = st.session_state[
            "raw_df"
        ].copy()  # this DF is used later for newsletter generation
        st.session_state["raw_df"] = (
            split_df_by_paragraphs(st.session_state["raw_df"])
            .drop("index", axis=1)
            .sort_values(
                by=TIMESTAMP_COLUMN,
                ascending=False,
            )
            .reset_index(drop=True)
            .reset_index()
        )

    # Clean dataset using min_text_length
    st.session_state["cleaned_df"] = clean_dataset(
        st.session_state["raw_df"],
        st.session_state["min_text_length"],
    )

    # Stop app if dataset is empty
    if st.session_state["cleaned_df"].empty:
        st.error("Not enough remaining data after cleaning", icon="🚨")
        st.stop()

    # Select time range
    min_max = st.session_state["cleaned_df"][TIMESTAMP_COLUMN].agg(["min", "max"])
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

    # Filter dataset to select only text within time range
    st.session_state["timefiltered_df"] = (
        st.session_state["cleaned_df"]
        .query(
            f"timestamp >= '{timestamp_range[0]}' and timestamp <= '{timestamp_range[1]}'"
        )
        .reset_index(drop=True)
    )
    
    ########## Remove duplicates from timefiltered_df, just in case of overlapping documents throughout different files ##########
    st.session_state["timefiltered_df"] = (st.session_state["timefiltered_df"].drop_duplicates(keep='first').reset_index(drop=True))
    st.session_state["timefiltered_df"]["index"] = st.session_state["timefiltered_df"].index
    st.write(f"Found {len(st.session_state['timefiltered_df'])} documents.")
    # print("DEBUG: ", st.session_state["timefiltered_df"].columns)





def train_model():
    ### TRAIN MODEL ###
    if parameters_sidebar_clicked:
        if "timefiltered_df" in st.session_state and not st.session_state["timefiltered_df"].empty:
            full_dataset = st.session_state["raw_df"]
            indices = st.session_state["timefiltered_df"]["index"]
                        
            (   st.session_state["topic_model"],
                st.session_state["topics"],
                _,
                st.session_state["embeddings"],
            ) = train_BERTopic_wrapper(
                dataset=full_dataset,
                indices=indices,
                form_parameters=st.session_state["parameters"],
                cache_base_name=st.session_state["data_name"]
                if not st.session_state["split_by_paragraphs"]
                else f'{st.session_state["data_name"]}_split_by_paragraphs',
            )
            
            st.session_state["topics_info"] = (
                st.session_state["topic_model"].get_topic_info().iloc[1:]
            )  # exclude -1 topic from topic list

            # Computes coherence value
            coherence_score_type = "c_npmi"
            coherence = get_coherence_value(
                st.session_state["topic_model"],
                st.session_state["topics"],
                st.session_state["timefiltered_df"][TEXT_COLUMN],
                coherence_score_type
            )
            diversity_score_type = "puw"
            diversity = get_diversity_value(st.session_state["topic_model"],
                                            st.session_state["topics"],
                                            st.session_state["timefiltered_df"][TEXT_COLUMN],
                                            diversity_score_type="puw")
            
            logger.info(f"Coherence score [{coherence_score_type}]: {coherence}")
            logger.info(f"Diversity score [{diversity_score_type}]: {diversity}")
            st.session_state['model_trained'] = True
            if not st.session_state['model_saved']: st.warning('Don\'t forget to save your model!', icon="⚠️")
            
        else:
            st.error("No data available for training. Please ensure data is correctly loaded.")



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
## MAIN PAGE
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

    parameters_sidebar_clicked = st.form_submit_button(
        "Train model", type="primary", on_click=save_widget_state
    )    
    
# Load selected DataFrame
select_data()

# Data overview
data_overview(st.session_state["timefiltered_df"])

# Train model
train_model()

# Save the model button
save_model_interface()



