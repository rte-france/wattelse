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
                        
            (
                st.session_state["topic_model"],
                st.session_state["topics"],
                _,
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
        else:
            st.error("No data available for training. Please ensure data is correctly loaded.")


def overall_results():
    if not ("topic_model" in st.session_state.keys()):
        st.stop()
    # Plot overall results
    try:
        with st.expander("Overall results"):
            st.write(
                plot_2d_topics(
                    st.session_state.parameters, st.session_state["topic_model"]
                )
            )
    except TypeError as te:  # we have sometimes: TypeError: Cannot use scipy.linalg.eigh for sparse A with k >= N. Use scipy.linalg.eigh(A.toarray()) or reduce k.
        logger.error(f"Error occurred: {te}")
        st.error("Cannot display overall results", icon="🚨")
        st.exception(te)
    except ValueError as ve:  # we have sometimes: ValueError: zero-size array to reduction operation maximum which has no identity
        logger.error(f"Error occurred: {ve}")
        st.error("Error computing overall results", icon="🚨")
        st.exception(ve)
        st.warning(f"Try to change the UMAP parameters", icon="⚠️")
        st.stop()


def dynamic_topic_modelling():
    with st.spinner("Computing topics over time..."):
        with st.expander("Dynamic topic modelling"):
            if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"].keys():
                st.write("## Dynamic topic modelling")

                # Parameters
                st.text_input(
                    "Topics list (format 1,12,52 or 1:20)",
                    key="dynamic_topics_list",
                    value="0:10",
                )
                st.number_input("nr_bins", min_value=1, value=10, key="nr_bins")

                # Compute topics over time only when train button is clicked
                if parameters_sidebar_clicked:
                    st.session_state["topics_over_time"] = compute_topics_over_time(
                        st.session_state["parameters"],
                        st.session_state["topic_model"],
                        st.session_state["timefiltered_df"],
                        nr_bins=st.session_state["nr_bins"],
                    )

                # Visualize
                st.write(
                    plot_topics_over_time(
                        st.session_state["topics_over_time"],
                        st.session_state["dynamic_topics_list"],
                        st.session_state["topic_model"],
                    )
                )
                



def create_treemap(topic_info_df):
    """
    Creates a treemap visualization of topics and their corresponding documents.

    Parameters:
    - topic_info_df: DataFrame with columns ['topic', 'number_of_documents', 'list_of_documents'].
    """

    labels = []  # Stores labels for topics and documents
    parents = []  # Stores the parent of each node (empty string for root nodes)
    values = []  # Stores values to control the size of each node

    for _, row in topic_info_df.iterrows():
        topic_label = f"{row['topic']} ({row['number_of_documents']})"
        labels.append(topic_label)
        parents.append("")
        values.append(row['number_of_documents'])

        for doc in row['list_of_documents']:
            labels.append(doc[:100])  # Truncate long documents for readability
            parents.append(topic_label)
            values.append(1)  # Assigning equal value to all documents for uniform size

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        textinfo="label+value",
        marker=dict(colors=[],
        line=dict(width=0),
        pad=dict(t=0)),
        # Here you can adjust the font size and family
        textfont=dict(
            size=20,  # Adjust the font size as needed
            family="Arial"  # Choose your desired font family
        )
    ))

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig






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
            except Exception as e:
                st.error(f"Failed to save the model: {e}")
        else:
            st.error("No model available to save. Please train a model first.")



def create_topic_info_dataframe(topic_model: BERTopic, docs: List[str], topic_assignments: List[int]) -> pd.DataFrame:
    """
    Create a DataFrame containing topics, the number of documents per topic, and the list of documents for each topic.

    Parameters:
    - topic_model: The BERTopic model from which topics are derived.
    - docs (List[str]): List of all documents.
    - topic_assignments (List[int]): List of topic assignments for each document.

    Returns:
    - DataFrame with columns ['topic', 'number_of_documents', 'list_of_documents'].
    """
    # Initialize the DataFrame
    topic_info = pd.DataFrame({"Document": docs, "Topic": topic_assignments})
    
    # Group by topic and aggregate information
    topic_info_agg = topic_info.groupby("Topic").agg({
        "Document": ["count", lambda x: list(x)]
    }).reset_index()

    # Rename columns for clarity
    topic_info_agg.columns = ["topic", "number_of_documents", "list_of_documents"]

    # Replace topic numbers with actual topic words, handling outliers if necessary
    topic_info_agg["topic"] = topic_info_agg["topic"].apply(
        lambda x: ", ".join([word for word, _ in topic_model.get_topic(x)]) if x != -1 else "Outlier"
    )

    return topic_info_agg





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

# Overall results
overall_results()

# Dynamic topic modelling
dynamic_topic_modelling()



#### FOR TREEMAP VISUALIZATION

# Extracting documents and their corresponding topic assignments
docs = st.session_state['timefiltered_df'][TEXT_COLUMN].tolist()
topic_assignments = st.session_state['topics']

# Create the topic info DataFrame
topic_info_df = create_topic_info_dataframe(st.session_state["topic_model"], docs, topic_assignments)

# Update session state with the DataFrame for later use
st.session_state["topic_info_df"] = topic_info_df
    

# Wrap the treemap in an expander
with st.expander("View Treemap Visualization", expanded=False):
    treemap_fig = create_treemap(st.session_state['topic_info_df'])
    st.plotly_chart(treemap_fig, use_container_width=True)

