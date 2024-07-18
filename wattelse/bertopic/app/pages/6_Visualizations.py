import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import locale
from loguru import logger
from umap import UMAP
import numpy as np
import datamapplot
from jinja2 import Template
import codecs
from sklearn.manifold import TSNE
import seaborn as sns

from wattelse.bertopic.utils import TIMESTAMP_COLUMN
from app_utils import plot_topics_over_time
from state_utils import restore_widget_state
from pathlib import Path
from wattelse.bertopic.app.app_utils import (
    plot_2d_topics,
    plot_topics_over_time,
    compute_topics_over_time,
)

import plotly.graph_objects as go

from wattelse.bertopic.utils import (
    TIMESTAMP_COLUMN,
    TEXT_COLUMN,
)

# Set locale to get french date names
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

# Restore widget state
restore_widget_state()

# Stop script if no model is trained
if "topic_model" not in st.session_state.keys():
    st.error("Train a model to explore different visualizations.", icon="🚨")
    st.stop()

####### Different visualization functions ########
def overall_results():
    if not ("topic_model" in st.session_state.keys()):
        st.stop()
    # Plot overall results
    try:
        with st.expander("Overall results"):
            st.plotly_chart(
                plot_2d_topics(
                    st.session_state.parameters, st.session_state["topic_model"]
                ), use_container_width=True)
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

def create_topic_info_dataframe():
    """
    Create a DataFrame containing topics, the number of documents per topic, and the list of documents for each topic.
    """
    # Extracting documents and their corresponding topic assignments
    docs = st.session_state['timefiltered_df'][TEXT_COLUMN].tolist()
    topic_assignments = st.session_state['topics']

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
        lambda x: ", ".join([word for word, _ in st.session_state['topic_model'].get_topic(x)]) if x != -1 else "Outlier"
    )

    topic_info_agg = topic_info_agg[topic_info_agg['topic'] != "Outlier"]
    # Update session state with the DataFrame for later use
    st.session_state["topic_info_df"] = topic_info_agg

def create_treemap():
    """
    Creates a treemap visualization of topics and their corresponding documents.
    """

    with st.spinner("Computing topics treemap..."):
        with st.expander("Topics Treemap"):
            create_topic_info_dataframe()
            
            labels = []  # Stores labels for topics and documents
            parents = []  # Stores the parent of each node (empty string for root nodes)
            values = []  # Stores values to control the size of each node

            for _, row in st.session_state['topic_info_df'].iterrows():
                topic_label = f"{row['topic']} ({row['number_of_documents']})"
                labels.append(topic_label)
                parents.append("")
                values.append(row['number_of_documents'])

                for doc in row['list_of_documents']:
                    labels.append(doc[:50]+'...')  # Truncate long documents for readability
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
                    size=34,  # Adjust the font size as needed
                )
            ))
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            fig.update_traces(marker=dict(cornerradius=10))
            st.plotly_chart(fig, use_container_width=True)

def calculate_document_lengths(documents):
    """
    Calculate the length of each document in terms of word count.
    """
    return np.array([len(doc.split()) for doc in documents])

def create_datamap():
    with st.spinner("Loading Data-map plot..."):
        # Calculate 2D embeddings
        reduced_embeddings = UMAP(n_neighbors=5, n_components=2, min_dist=0.2, metric='cosine').fit_transform(st.session_state["embeddings"])

        # Create a dataframe that associates documents with their embeddings and the topics they belong to
        topic_nums = list(set(st.session_state['topics']))

        topic_info = st.session_state['topic_model'].get_topic_info()
        topic_representations = {row['Topic']: row['Name'] for index, row in topic_info.iterrows() if row['Topic'] in topic_nums}
        docs = st.session_state['timefiltered_df'][TEXT_COLUMN].tolist()
        data = {
            "document": docs,
            "embedding": list(reduced_embeddings),
            "topic_num": st.session_state['topics'],
            "topic_representation": [topic_representations[topic] if topic in topic_representations else f"-1_{topic}" for topic in st.session_state['topics']]
        }
        df = pd.DataFrame(data)

        # Add option to include or exclude noise points
        include_noise = st.sidebar.checkbox("Include unlabelled text (Topic = -1)", value=False)
        if not include_noise:
            df = df[~df['topic_representation'].str.contains('-1')]

        # Treat topics starting with -1_ as noise
        df['is_noise'] = df['topic_representation'].str.contains('-1')
        df['topic_representation'] = df.apply(lambda row: "" if row['is_noise'] else topic_representations.get(row['topic_num'], ""), axis=1)
        df['topic_color'] = df.apply(lambda row: '#999999' if row['is_noise'] else None, axis=1)

        # Ensure we have at least one valid topic
        if df.empty:
            st.warning("No valid topics to visualize. All documents might be classified as outliers.")
            return

        # Prepare the data for datamapplot (conversion to numpy arrays)
        embeddings_array = np.array(df['embedding'].tolist())
        topic_representations_array = df['topic_representation'].values

        # Handle color mapping for topics
        unique_topics = df['topic_representation'].unique()
        unique_topics = unique_topics[unique_topics != ""]
        color_palette = sns.color_palette("tab20", len(unique_topics)).as_hex()
        color_mapping = {topic: color for topic, color in zip(unique_topics, color_palette)}
        df['topic_color'] = df.apply(lambda row: color_mapping.get(row['topic_representation'], row['topic_color']), axis=1)

        try:
            hover_data = df['document'].tolist()
            document_lengths = calculate_document_lengths(hover_data)
            # Normalize the sizes (optional)
            min_size = 5
            max_size = 50
            normalized_sizes = min_size + (max_size - min_size) * (document_lengths - document_lengths.min()) / (document_lengths.max() - document_lengths.min())

            plot = datamapplot.create_interactive_plot(
                embeddings_array,
                topic_representations_array,
                hover_text=hover_data,
                marker_size_array=normalized_sizes,
                inline_data=True,
                noise_label="",  # Setting noise_label to an empty string
                noise_color="#999999",  # Setting noise_color to grey
                color_label_text=True,
                label_wrap_width=16,
                label_color_map=color_mapping,
                width="100%",
                height="100%",
                darkmode=False,
                marker_color_array=df['topic_color'].values,
                use_medoids=True,
                cluster_boundary_polygons=False,  # Keeping it False as per your request
                enable_search=True,
                search_field="hover_text",
                point_line_width=0,
                logo='https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/RTE_logo.svg/1024px-RTE_logo.svg.png',
                logo_width=100,
            )

            # Encode the HTML string using UTF-8
            html_str = plot._html_str.encode(encoding='UTF-8', errors='replace')
            save_path = Path(__file__).parent.parent / 'datamapplot.html'
            with open(save_path, 'wb') as f:
                f.write(html_str)

            HtmlFile = open(save_path, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, width=1500, height=1000, scrolling=True)

        except KeyError as e:
            st.error(f"An error occurred while creating the datamap: {str(e)}")
            st.warning("This might be due to empty clusters or insufficient data for visualization.")
            logger.error(f"Datamap creation error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.error(f"Unexpected error in datamap creation: {str(e)}")

################################################
## VIZ PAGE
################################################

# Wide layout
st.set_page_config(page_title="Wattelse® topic", layout="wide")

# Restore widget state
restore_widget_state()

### TITLE ###
st.title("Visualizations")

# # Overall results
# overall_results()

# # For treemap
# create_treemap()

# For datamap
create_datamap()
