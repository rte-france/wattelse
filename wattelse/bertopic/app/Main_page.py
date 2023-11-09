import streamlit as st
from loguru import logger
from wattelse.bertopic.utils import (
    TIMESTAMP_COLUMN,
    DATA_DIR,
    clean_dataset,
    split_df_by_paragraphs,
)

from wattelse.bertopic.app.state_utils import register_widget, save_widget_state, restore_widget_state

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
    plot_docs_reparition_over_time,
    initialize_default_parameters_keys,
    load_data_wrapper
)
from wattelse.bertopic.app.train_utils import train_BERTopic_wrapper


def reset_topics():
    # shall be called when we update data parameters (timestamp, min char, split, etc.)
    st.session_state.pop("selected_topic_number", None)
    st.session_state.pop("new_topics", None)
    st.session_state.pop("new_topics_over_time", None)
    save_widget_state()


def reset_all():
    # TODO: add here all state variables we want to reset when we change the data
    st.session_state.pop("timestamp_range", None)
    reset_topics()

def reset_data():
    # TODO: add here all state variables we want to reset when we change the data
    st.session_state.pop("timestamp_range", None)
    st.session_state.pop("data_name", None)
    reset_topics()

def select_data():
    st.write("## Data selection")

    if "data_folder" not in st.session_state:
        st.session_state["data_folder"] = DATA_DIR

    data_folders = sorted(
        set(
            f.parent
            for f in list(DATA_DIR.glob("**/*.csv")) + list(DATA_DIR.glob("**/*.jsonl"))
        )
    )
    data_options = ["None"] + sorted(
        [
            p.name
            for p in list(st.session_state["data_folder"].glob("*.csv"))
            + list(st.session_state["data_folder"].glob("*.jsonl*"))
        ]
    )

    # Select box with every file saved in DATA_DIR as options
    if "data_name" not in st.session_state:
        st.session_state["data_name"] = data_options[0]

    register_widget("data_name")
    register_widget("data_folder")
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.selectbox(
            "Base folder", data_folders, key="data_folder", on_change=reset_data
        )
    with col2:
        st.selectbox(
            "Select data to continue",
            data_options,
            key="data_name",
            on_change=reset_all,
        )

    # Stop the app as long as no data is selected
    if st.session_state["data_name"] == "None":
        st.stop()

    st.session_state["raw_df"] = (
        load_data_wrapper(
            f"{st.session_state['data_folder']}/{st.session_state['data_name']}"
        )
        .sort_values(by=TIMESTAMP_COLUMN, ascending=False)
        .reset_index(drop=True)
        .reset_index()
    )

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
    st.write(f"Found {len(st.session_state['timefiltered_df'])} documents.")


def data_overview():
    with st.expander("Data overview"):
        freq = st.select_slider(
            "Time aggregation",
            options=(
                "1D",
                "2D",
                "1W",
                "2W",
                "1M",
                "2M",
                "1Y",
                "2Y",
            ),
            value="1M",
        )
        plot_docs_reparition_over_time(st.session_state["timefiltered_df"], freq)


def train_model():
    ### TRAIN MODEL ###
    if parameters_sidebar_clicked:
        # Train
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


################################################
## MAIN PAGE
################################################

# Wide layout
st.set_page_config(layout="wide")

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
data_overview()

# Train model
train_model()

# Overall results
overall_results()

# Dynamic topic modelling
dynamic_topic_modelling()
