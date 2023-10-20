import streamlit as st

from wattelse.bertopic.utils import TIMESTAMP_COLUMN
from app_utils import print_docs_for_specific_topic, plot_topics_over_time, generate_newsletter, load_data
from state_utils import register_widget, save_widget_state, restore_widget_state

# Restore widget state
restore_widget_state()

# Stop script if no model is trained
if "topic_model" not in st.session_state.keys():
	st.error("Train a model to explore generated topics.", icon="🚨")
	st.stop()

### SIDEBAR ###

def set_topic_selection(selected_topic_number):
	st.session_state["selected_topic_number"] = selected_topic_number

def find_similar_topic():
	similar_topics, _ = st.session_state["topic_model"].find_topics(st.session_state["search_terms"], top_n=1)
	st.session_state["selected_topic_number"] = similar_topics[0]


with st.sidebar:

	# Automatic newsletter
	if st.button("Generate newsletter"):
		generate_newsletter(st.session_state["topic_model"], load_data(st.session_state["data_name"]), st.session_state["topics"], df_split = st.session_state["timefiltered_df"])

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

st.write(f"# Topic {st.session_state['selected_topic_number']} : {topic_docs_number} documents")


st.markdown(f"## {' | '.join(topic_words)}")

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

# Show documents belonging to the topic
st.write("## Documents")
print_docs_for_specific_topic(st.session_state["timefiltered_df"], st.session_state["topics"], st.session_state["selected_topic_number"])

# If remaining data is processed, print new docs
if "new_topics_over_time" in st.session_state.keys():
	st.write("## New documents")
	print_docs_for_specific_topic(st.session_state["remaining_df"], st.session_state["new_topics"], st.session_state["selected_topic_number"])