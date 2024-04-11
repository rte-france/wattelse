from statistics import StatisticsError

import streamlit as st

from wattelse.bertopic.metrics import TopicMetrics, TIME_WEIGHT
from state_utils import register_widget, save_widget_state, restore_widget_state
from wattelse.bertopic.utils import TIMESTAMP_COLUMN
from wattelse.bertopic.app.app_utils import compute_topics_over_time

# Restore widget state
restore_widget_state()

# Stop script if no model is trained
if "topic_model" not in st.session_state.keys():
    st.error("Train a model to explore generated topics.", icon="🚨")
    st.stop()

if "topics_over_time"  not in st.session_state.keys():
    if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"].keys():
        st.session_state["topics_over_time"] = compute_topics_over_time(
            st.session_state["parameters"],
            st.session_state["topic_model"],
            st.session_state["timefiltered_df"],
            nr_bins=10,
        )

if "tw" not in st.session_state.keys():
    st.session_state["tw"] = TIME_WEIGHT

### TITLE ###
st.title("Topics map")

register_widget("tw")
st.slider("Time weight", min_value=0.0, max_value=0.1, step=0.005, key="tw", on_change=save_widget_state)


### Main page ###
topic_metrics = TopicMetrics(st.session_state["topic_model"], st.session_state["topics_over_time"])
try:
    st.plotly_chart(topic_metrics.plot_TEM_map(st.session_state["tw"]), use_container_width=True)
except StatisticsError as se:
    st.warning(f"Try to change the Time Weight value: {se}", icon="⚠️")
    st.stop()


#save_state()