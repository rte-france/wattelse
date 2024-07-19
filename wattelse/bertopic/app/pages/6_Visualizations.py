import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import locale
from loguru import logger
from umap import UMAP
import datamapplot
from pathlib import Path
import seaborn as sns
import plotly.graph_objects as go

from wattelse.bertopic.utils import TIMESTAMP_COLUMN, TEXT_COLUMN
from app_utils import plot_topics_over_time
from state_utils import restore_widget_state
from wattelse.bertopic.app.app_utils import (
    plot_2d_topics,
    plot_topics_over_time,
    compute_topics_over_time,
)

# Set locale for French date names
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

# Restore widget state and set up page
restore_widget_state()
st.set_page_config(page_title="Wattelse® topic", layout="wide")

# Check if a model is trained
if "topic_model" not in st.session_state:
    st.error("Train a model to explore different visualizations.", icon="🚨")
    st.stop()

def overall_results():
    """Display overall results visualization."""
    try:
        with st.expander("Overall results"):
            st.plotly_chart(
                plot_2d_topics(
                    st.session_state.parameters, st.session_state["topic_model"]
                ), use_container_width=True)
    except TypeError as te:
        logger.error(f"Error occurred: {te}")
        st.error("Cannot display overall results", icon="🚨")
        st.exception(te)
    except ValueError as ve:
        logger.error(f"Error occurred: {ve}")
        st.error("Error computing overall results", icon="🚨")
        st.exception(ve)
        st.warning("Try to change the UMAP parameters", icon="⚠️")
        st.stop()

def create_topic_info_dataframe():
    """Create a DataFrame containing topics, number of documents per topic, and list of documents for each topic."""
    docs = st.session_state['timefiltered_df'][TEXT_COLUMN].tolist()
    topic_assignments = st.session_state['topics']

    topic_info = pd.DataFrame({"Document": docs, "Topic": topic_assignments})
    topic_info_agg = topic_info.groupby("Topic").agg({
        "Document": ["count", list]
    }).reset_index()

    topic_info_agg.columns = ["topic", "number_of_documents", "list_of_documents"]
    topic_info_agg["topic"] = topic_info_agg["topic"].apply(
        lambda x: ", ".join([word for word, _ in st.session_state['topic_model'].get_topic(x)]) if x != -1 else "Outlier"
    )

    st.session_state["topic_info_df"] = topic_info_agg[topic_info_agg['topic'] != "Outlier"]

def create_treemap():
    """Create a treemap visualization of topics and their corresponding documents."""
    with st.spinner("Computing topics treemap..."):
        with st.expander("Topics Treemap"):
            create_topic_info_dataframe()
            
            labels, parents, values = [], [], []
            for _, row in st.session_state['topic_info_df'].iterrows():
                topic_label = f"{row['topic']} ({row['number_of_documents']})"
                labels.append(topic_label)
                parents.append("")
                values.append(row['number_of_documents'])

                for doc in row['list_of_documents']:
                    labels.append(doc[:50]+'...')
                    parents.append(topic_label)
                    values.append(1)

            fig = go.Figure(go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                textinfo="label+value",
                marker=dict(colors=[], line=dict(width=0), pad=dict(t=0)),
                textfont=dict(size=34)
            ))
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            fig.update_traces(marker=dict(cornerradius=10))
            st.plotly_chart(fig, use_container_width=True)

def calculate_document_lengths(documents):
    """Calculate the length of each document in terms of word count."""
    return np.array([len(doc.split()) for doc in documents])

def create_datamap():
    """Create an interactive data map visualization."""
    with st.spinner("Loading Data-map plot..."):
        reduced_embeddings = UMAP(n_neighbors=5, n_components=2, min_dist=0.2, metric='cosine').fit_transform(st.session_state["embeddings"])

        topic_nums = list(set(st.session_state['topics']))
        topic_info = st.session_state['topic_model'].get_topic_info()
        topic_representations = {row['Topic']: row['Name'] for _, row in topic_info.iterrows() if row['Topic'] in topic_nums}
        docs = st.session_state['timefiltered_df'][TEXT_COLUMN].tolist()
        
        df = pd.DataFrame({
            "document": docs,
            "embedding": list(reduced_embeddings),
            "topic_num": st.session_state['topics'],
            "topic_representation": [topic_representations.get(topic, f"-1_{topic}") for topic in st.session_state['topics']]
        })

        include_noise = st.sidebar.checkbox("Include unlabelled text (Topic = -1)", value=False)
        if not include_noise:
            df = df[~df['topic_representation'].str.contains('-1')]

        df['is_noise'] = df['topic_representation'].str.contains('-1')
        df['topic_representation'] = df.apply(lambda row: "" if row['is_noise'] else topic_representations.get(row['topic_num'], ""), axis=1)
        df['topic_color'] = df.apply(lambda row: '#999999' if row['is_noise'] else None, axis=1)

        if df.empty:
            st.warning("No valid topics to visualize. All documents might be classified as outliers.")
            return

        embeddings_array = np.array(df['embedding'].tolist())
        topic_representations_array = df['topic_representation'].values

        unique_topics = df['topic_representation'].unique()
        unique_topics = unique_topics[unique_topics != ""]
        color_palette = sns.color_palette("tab20", len(unique_topics)).as_hex()
        color_mapping = dict(zip(unique_topics, color_palette))
        df['topic_color'] = df.apply(lambda row: color_mapping.get(row['topic_representation'], row['topic_color']), axis=1)

        try:
            hover_data = df['document'].tolist()
            document_lengths = calculate_document_lengths(hover_data)
            normalized_sizes = 5 + 45 * (document_lengths - document_lengths.min()) / (document_lengths.max() - document_lengths.min())

            plot = datamapplot.create_interactive_plot(
                embeddings_array,
                topic_representations_array,
                hover_text=hover_data,
                marker_size_array=normalized_sizes,
                inline_data=True,
                noise_label="",
                noise_color="#999999",
                color_label_text=True,
                label_wrap_width=16,
                label_color_map=color_mapping,
                width="100%",
                height="100%",
                darkmode=False,
                marker_color_array=df['topic_color'].values,
                use_medoids=True,
                cluster_boundary_polygons=False,
                enable_search=True,
                search_field="hover_text",
                point_line_width=0,
                logo='https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/RTE_logo.svg/1024px-RTE_logo.svg.png',
                logo_width=100,
            )

            save_path = Path(__file__).parent.parent / 'datamapplot.html'
            with open(save_path, 'wb') as f:
                f.write(plot._html_str.encode(encoding='UTF-8', errors='replace'))

            with open(save_path, 'r', encoding='utf-8') as HtmlFile:
                source_code = HtmlFile.read()
            components.html(source_code, width=1500, height=1000, scrolling=True)

        except KeyError as e:
            st.error(f"An error occurred while creating the datamap: {str(e)}")
            st.warning("This might be due to empty clusters or insufficient data for visualization.")
            logger.error(f"Datamap creation error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.error(f"Unexpected error in datamap creation: {str(e)}")

# Main execution
st.title("Visualizations")

# Uncomment these lines if you want to include these visualizations
# overall_results()
# create_treemap()

create_datamap()

# FIXME: cluster_boundary_polygons=True causes a "pop from an empty set" error in the data map plot's generation process. 
# It's not urgent, but should be looked into to see what's causing the problem and potentially get a better visualization where clusters are delimitted with contours.