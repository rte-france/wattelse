#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import streamlit as st
import pandas as pd
from datetime import datetime
import locale

from urllib.parse import urlparse

from wattelse.bertopic.utils import TIMESTAMP_COLUMN
from wattelse.bertopic.newsletter_features import get_most_representative_docs
from app_utils import print_docs_for_specific_topic, plot_topics_over_time, load_data
from state_utils import register_widget, save_widget_state, restore_widget_state
from wattelse.bertopic.app.app_utils import compute_topics_over_time

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
		button_title = f"{topic_number+1} - {' | '.join(topic_words)}"
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

st.write(f"# Thème {st.session_state['selected_topic_number']+1} : {topic_docs_number} documents")


st.markdown(f"## #{' #'.join(topic_words)}")

# Plot topic over time
if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"].keys():
	# If remaining data not processed yet, print filtered_df topics over time
	if not "new_topics_over_time" in st.session_state.keys():
		st.write(
			plot_topics_over_time(
				st.session_state["topics_over_time"],
				str(st.session_state["selected_topic_number"]),
				st.session_state["topic_model"],
				)
			)
	# If remaining data is processed, add new data information to plot
	else:
		st.write(
			plot_topics_over_time(
				st.session_state["new_topics_over_time"],
				str(st.session_state["selected_topic_number"]),
				st.session_state["topic_model"],
				time_split=st.session_state["timefiltered_df"][TIMESTAMP_COLUMN].max(),
				)
			)

# Show most representative documents belonging to the topic
st.write("## Sélection d'articles représentatifs")

if st.session_state["split_by_paragraphs"]:
	representative_df = get_most_representative_docs(st.session_state["topic_model"],
												     st.session_state["initial_df"],
												     st.session_state["topics"],
													 df_split=st.session_state["timefiltered_df"],
												     topic_number=st.session_state["selected_topic_number"],
													 top_n_docs=6,
													 )
else: # use both cluster_probability and ctfidf_representation modes
	representative_df_1 = get_most_representative_docs(st.session_state["topic_model"],
													st.session_state["timefiltered_df"],
													st.session_state["topics"],
													topic_number=st.session_state["selected_topic_number"],
													mode="cluster_probability",
													top_n_docs=3,
													)
	representative_df_2 = get_most_representative_docs(st.session_state["topic_model"],
													st.session_state["timefiltered_df"],
													st.session_state["topics"],
													topic_number=st.session_state["selected_topic_number"],
													mode="ctfidf_representation",
													top_n_docs=3,
													)
	representative_df = pd.concat([representative_df_1, representative_df_2]).drop_duplicates()


representative_df = representative_df.sort_values(by="timestamp", ascending=False) # sort values by date

col1, col2 = st.columns([2,1]) # set button max size using columns
with col1:
	for i, doc in representative_df.iterrows(): # write list of representative docs
		website_name = urlparse(doc.url).netloc.replace("www.","").split(".")[0]
		date = doc.timestamp.strftime("%A %d %b %Y")
		st.link_button(f"*{website_name}*\n\n**{doc.title}**\n\n{date}", doc.url)


# If remaining data is processed, print new docs
if "new_topics_over_time" in st.session_state.keys():
	st.write("## New documents")
	print_docs_for_specific_topic(st.session_state["remaining_df"], st.session_state["new_topics"], st.session_state["selected_topic_number"])