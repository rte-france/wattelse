import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import locale
from loguru import logger
from umap import UMAP
import numpy as np
import datamapplot
from jinja2 import Template


from wattelse.bertopic.utils import TIMESTAMP_COLUMN
from app_utils import plot_topics_over_time
from state_utils import restore_widget_state

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
                # st.number_input("nr_bins", min_value=1, value=10, key="nr_bins")

                # Compute topics over time only when train button is clicked
                st.session_state["topics_over_time"] = compute_topics_over_time(
                    st.session_state["parameters"],
                    st.session_state["topic_model"],
                    st.session_state["timefiltered_df"],
                    nr_bins=10,
                )

                # Visualize
                st.plotly_chart(plot_topics_over_time(
                        st.session_state["topics_over_time"],
                        st.session_state["dynamic_topics_list"],
                        st.session_state["topic_model"],
                    ), use_container_width=True)

                

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


def create_datamap():
    with st.spinner("Loading Data-map plot..."):
        # Calculate 2D embeddings
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', verbose=True).fit_transform(st.session_state["embeddings"])
        
        # Create a dataframe that associates documents with their embeddings and the topics they belong to
        topic_nums = list(set(st.session_state['topics']))
        topic_info = st.session_state['topic_model'].get_topic_info()  
        topic_representations = {row['Topic']: row['Name'] for index, row in topic_info.iterrows() if row['Topic'] in topic_nums}
        docs = st.session_state['timefiltered_df'][TEXT_COLUMN].tolist()
                
        data = {
            "document": docs,
            "embedding": list(reduced_embeddings),
            "topic_num": st.session_state['topics'],
            "topic_representation": [topic_representations[topic] for topic in st.session_state['topics']]
        }
        df = pd.DataFrame(data)
        
        # Prepare the data for datamapplot (conversion to numpy arrays)
        embeddings_array = np.array(df['embedding'].tolist())

        df.loc[df['topic_num'] == -1, 'topic_representation'] = 'Unlabeled'

        # Convert the topic_representation column to a NumPy array
        topic_representations_array = df['topic_representation'].values
        plot = datamapplot.create_interactive_plot(
                embeddings_array,
                topic_representations_array,
                hover_text = docs,
                enable_search=True,
                darkmode=False,
                noise_color="#aaaaaa44",
                logo='https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/RTE_logo.svg/1024px-RTE_logo.svg.png',
                logo_width=80,
            )
        
        plot.save('datamapplot.html')
        HtmlFile = open("datamapplot.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, width=1200, height=800)
        





            

    


################################################
## VIZ PAGE
################################################



# Wide layout
st.set_page_config(page_title="Wattelse® topic", layout="wide")

# Restore widget state
restore_widget_state()


### TITLE ###
st.title("Visualizations")

# Overall results
# overall_results()

# Dynamic topic modelling
dynamic_topic_modelling()

# For treemap
create_treemap()

# For datamap
create_datamap()
