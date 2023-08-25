import streamlit as st
import pandas as pd
from utils import TIMESTAMP_COLUMN
from app_utils import print_docs_for_specific_topic

# Stop script if no model is trained
if "topic_model" not in st.session_state.keys():
	st.write("Train a model to explore generated topics.")
	st.stop()


### TITLE ###

st.title("Topics exploration")


### SIDEBAR ###

def set_topic_selection(selected_topic_number):
	st.session_state["selected_topic_number"] = selected_topic_number

with st.sidebar:
	for index, topic in st.session_state["topics_list"].iterrows():
		topic_number = topic["Topic"]
		topic_words = topic["Representation"][:3]
		st.button(f"{topic_number} - {' | '.join(topic_words)}", use_container_width=True, on_click=set_topic_selection, args=(topic_number,))



### PAGE ###

# Stop app if no topic is selected
if "selected_topic_number" not in st.session_state.keys():
	st.stop()

# Print selected topic and topic stats/info
topic_docs_number = st.session_state["topics_list"].iloc[st.session_state["selected_topic_number"]]["Count"]
topic_words = st.session_state["topics_list"].iloc[st.session_state["selected_topic_number"]]["Representation"]

st.write(f"## Topic {st.session_state['selected_topic_number']} : {topic_docs_number} documents")
st.markdown(f"### {' | '.join(topic_words)}")

# Plot topic over time
if TIMESTAMP_COLUMN in st.session_state["df"].keys():
	st.write(st.session_state["topic_model"].visualize_topics_over_time(st.session_state["topics_over_time"], topics=[st.session_state["selected_topic_number"]], width=700))

# Show documents belonging to the topic

print_docs_for_specific_topic(st.session_state["df"], st.session_state["topic_per_doc"], st.session_state["selected_topic_number"])