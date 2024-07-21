import streamlit as st
from statistics import StatisticsError

from wattelse.bertopic.metrics import TopicMetrics, TIME_WEIGHT
from state_utils import register_widget, save_widget_state, restore_widget_state
from wattelse.bertopic.utils import TIMESTAMP_COLUMN
from wattelse.bertopic.app.app_utils import compute_topics_over_time

# Restore widget state
restore_widget_state()

# Check if a model is trained
if "topic_model" not in st.session_state:
    st.error("Train a model to explore generated topics.", icon="🚨")
    st.stop()

# Compute topics over time if not already done
if "topics_over_time" not in st.session_state and TIMESTAMP_COLUMN in st.session_state["timefiltered_df"]:
    st.session_state["topics_over_time"] = compute_topics_over_time(
        st.session_state["parameters"],
        st.session_state["topic_model"],
        st.session_state["timefiltered_df"],
        nr_bins=10,
    )

# Initialize time weight if not present
if "tw" not in st.session_state:
    st.session_state["tw"] = TIME_WEIGHT

# Title
st.title("Topics map")

# Time weight slider
register_widget("tw")
st.slider("Time weight", min_value=0.0, max_value=0.1, step=0.005, key="tw", on_change=save_widget_state)

# Main visualization
topic_metrics = TopicMetrics(st.session_state["topic_model"], st.session_state["topics_over_time"])
try:
    st.plotly_chart(topic_metrics.plot_TEM_map(st.session_state["tw"]), use_container_width=True)
except StatisticsError as se:
    st.warning(f"Try to change the Time Weight value: {se}", icon="⚠️")
    st.stop()