import streamlit as st
import pandas as pd
from datetime import datetime
import locale

from urllib.parse import urlparse
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from wattelse.bertopic.utils import TIMESTAMP_COLUMN
from wattelse.bertopic.newsletter_features import get_most_representative_docs
from app_utils import print_docs_for_specific_topic, plot_topics_over_time, load_data
from state_utils import register_widget, save_widget_state, restore_widget_state
from wattelse.bertopic.app.app_utils import compute_topics_over_time

import os
from datetime import timedelta

from jinja2 import Template
import html
import re

EXPORT_BASE_FOLDER = os.path.join('wattelse', 'bertopic', 'app', 'exported_topics')

def export_topic_documents(topic_docs, granularity_days, export_base_folder, topic_number, topic_words):
    """
    Export documents for a specific topic grouped by a given time granularity.
    
    Args:
    topic_docs (pd.DataFrame): DataFrame containing documents and timestamps for a specific topic.
    granularity_days (int): Number of days to group documents by.
    export_base_folder (str): Base folder path to export the text files.
    topic_number (int): The number of the current topic.
    topic_words (list): List of words representing the topic.
    
    Returns:
    str: Path to the created folder.
    """
    # Create a folder name based on topic number, first three words, and granularity
    folder_name = f"topic_{topic_number}_{' '.join(topic_words[:3])}_{granularity_days}days"
    folder_name = re.sub(r'[^\w\-_\. ]', '_', folder_name)  # Replace invalid characters
    export_folder = os.path.join(export_base_folder, folder_name)
    os.makedirs(export_folder, exist_ok=True)
    
    topic_docs = topic_docs.sort_values('timestamp')
    start_date = topic_docs['timestamp'].min()
    end_date = topic_docs['timestamp'].max()
    current_date = start_date
    
    while current_date <= end_date:
        next_date = current_date + timedelta(days=granularity_days)
        
        period_docs = topic_docs[(topic_docs['timestamp'] >= current_date) & (topic_docs['timestamp'] < next_date)]
        
        if not period_docs.empty:
            file_name = f"{current_date.strftime('%Y%m%d')}-{(next_date - timedelta(days=1)).strftime('%Y%m%d')}.txt"
            file_path = os.path.join(export_folder, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for _, doc in period_docs.iterrows():
                    f.write(f"{doc['text']}\n\n")  # Write raw document text with double line break
        
        current_date = next_date
    
    return export_folder

# Set locale to get french date names
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

# Restore widget state
restore_widget_state()

# Stop script if no model is trained
if "topic_model" not in st.session_state.keys():
	st.error("Train a model to explore generated topics.", icon="🚨")
	st.stop()
 

# Prepare topics over time if not done already
if "topics_over_time" not in st.session_state.keys():
	if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"].keys():
		# Compute topics over time only when train button is clicked
		st.session_state["topics_over_time"] = compute_topics_over_time(
			st.session_state["parameters"],
			st.session_state["topic_model"],
			st.session_state["timefiltered_df"],
			nr_bins=10,
		)
			

### SIDEBAR ###

def set_topic_selection(selected_topic_number):
	st.session_state["selected_topic_number"] = selected_topic_number

def find_similar_topic():
	similar_topics, _ = st.session_state["topic_model"].find_topics(st.session_state["search_terms"], top_n=1)
	st.session_state["selected_topic_number"] = similar_topics[0]


with st.sidebar:

    # Search bar
    search_terms = st.text_input("Search topic", on_change=find_similar_topic, key="search_terms")

    # Topics list
    for index, topic in st.session_state["topics_info"].iterrows():
        topic_number = topic["Topic"]
        topic_words = topic["Representation"][:3]
        button_title = f"{topic_number} - {' | '.join(topic_words)}"
        if "new_topics" in st.session_state.keys():
            new_docs_number = st.session_state["new_topics"].count(topic_number)
            button_title += f" :red[+{new_docs_number}]"
        st.button(button_title, use_container_width=True, on_click=set_topic_selection, args=(topic_number,))



### PAGE ###

# Stop app if no topic is selected
if "selected_topic_number" not in st.session_state.keys():
	st.stop()

# Print selected topic and topic stats/info
topic_docs_number = st.session_state["topics_info"].iloc[st.session_state["selected_topic_number"]]["Count"]
topic_words = st.session_state["topics_info"].iloc[st.session_state["selected_topic_number"]]["Representation"]

st.write(f"# Thème {st.session_state['selected_topic_number']} : {topic_docs_number} documents")


st.markdown(f"## #{' #'.join(topic_words)}")

# Plot topic over time
if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"].keys():
	# If remaining data not processed yet, print filtered_df topics over time
	if not "new_topics_over_time" in st.session_state.keys():
		st.plotly_chart(
			plot_topics_over_time(
				st.session_state["topics_over_time"],
				str(st.session_state["selected_topic_number"]),
				st.session_state["topic_model"],
				),
				use_container_width=True,
			)
	# If remaining data is processed, add new data information to plot
	else:
		st.plotly_chart(
			plot_topics_over_time(
				st.session_state["new_topics_over_time"],
				str(st.session_state["selected_topic_number"]),
				st.session_state["topic_model"],
				time_split=st.session_state["timefiltered_df"][TIMESTAMP_COLUMN].max(),
				),
				use_container_width=True,
			)


# Add a number input for top_n_docs selection
top_n_docs = st.number_input("Nombre d'articles à afficher", min_value=1, max_value=topic_docs_number, value=topic_docs_number, step=1)

# Determine which method to use based on whether df is split
if st.session_state["split_by_paragraphs"] == True:
    representative_df = get_most_representative_docs(
        st.session_state["topic_model"],
        st.session_state["initial_df"],
        st.session_state["topics"],
        df_split=st.session_state["timefiltered_df"],
        topic_number=st.session_state["selected_topic_number"],
        top_n_docs=top_n_docs,
    )
else:
    # Use both cluster_probability and ctfidf_representation modes
    representative_df_1 = get_most_representative_docs(
        st.session_state["topic_model"],
        st.session_state["timefiltered_df"],
        st.session_state["topics"],
        topic_number=st.session_state["selected_topic_number"],
        mode="cluster_probability",
        top_n_docs=top_n_docs,
    )
    representative_df_2 = get_most_representative_docs(
        st.session_state["topic_model"],
        st.session_state["timefiltered_df"],
        st.session_state["topics"],
        topic_number=st.session_state["selected_topic_number"],
        mode="ctfidf_representation",
        top_n_docs=top_n_docs,
    )
    representative_df = pd.concat([representative_df_1, representative_df_2]).drop_duplicates()

representative_df = representative_df.sort_values(by="timestamp", ascending=False)

# Create two columns
col1, col2 = st.columns([0.5, 0.5])

# Get unique sources
sources = representative_df['url'].apply(lambda x: urlparse(x).netloc.replace("www.", "").split(".")[0]).unique()

# Multi-select for sources
selected_sources = st.multiselect(
    "Sélectionnez les sources à afficher",
    options=['Tous'] + list(sources),
    default=['Tous']
)

# Filter the dataframe based on selected sources
if 'Tous' not in selected_sources:
    filtered_df = representative_df[representative_df['url'].apply(
        lambda x: urlparse(x).netloc.replace("www.", "").split(".")[0] in selected_sources
    )]
else:
    filtered_df = representative_df

with col1:
    # Pie chart of sources
    source_counts = representative_df['url'].apply(lambda x: urlparse(x).netloc.replace("www.", "").split(".")[0]).value_counts()
    
    # Create a list for the 'pull' parameter
    pull = [0.2 if source in selected_sources and 'Tous' not in selected_sources else 0 for source in source_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=source_counts.index,
        values=source_counts.values,
        pull=pull,
        textposition='inside',
        textinfo='percent+label',
        hole=0.3,
    )])
    
    fig.update_layout(
        showlegend=False,
        height=600,  # Increase the height of the chart
        width=500,   # Increase the width of the chart
        margin=dict(t=0, b=0, l=0, r=0)  # Reduce margins to maximize chart size
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:

    # Container for representative documents
    with st.container(border=False,height=600):
        for i, doc in filtered_df.iterrows():
            website_name = urlparse(doc.url).netloc.replace("www.", "").split(".")[0]
            date = doc.timestamp.strftime("%A %d %b %Y")
            
            # Create a snippet of the text content
            snippet = doc.text[:150] + "..." if len(doc.text) > 150 else doc.text
            
            # Use link_button with additional content
            st.link_button(
                f"*{website_name}*\n\n**{doc.title}**\n\n{date}\n\n{snippet}",
                doc.url
            )
            # Add a small space between buttons
            st.write("")

# If remaining data is processed, print new docs
if "new_topics_over_time" in st.session_state.keys():
	st.write("## New documents")
	print_docs_for_specific_topic(st.session_state["remaining_df"], st.session_state["new_topics"], st.session_state["selected_topic_number"])
      

# Add export functionality
st.write("## Export Topic Documents")
granularity_days = st.number_input("Granularity (number of days)", min_value=1, value=3, step=1)

if st.button("Export Topic Documents"):
    # Use the filtered_df that's already being used to display documents
    topic_docs = filtered_df.sort_values(by='timestamp')
    
    if not topic_docs.empty:
        topic_number = st.session_state["selected_topic_number"]
        topic_words = st.session_state["topic_model"].get_topic(topic_number)
        topic_words = [word for word, _ in topic_words]  # Extract just the words
        
        exported_folder = export_topic_documents(
            topic_docs, 
            granularity_days, 
            EXPORT_BASE_FOLDER,
            topic_number,
            topic_words
        )
        
        st.success(f"Successfully exported documents to folder: {exported_folder}")
        st.write("You can find the exported files in this folder.")
    else:
        st.warning("No documents found for the selected topic.")



