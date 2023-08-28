from statistics import StatisticsError

import streamlit as st

from metrics import TopicMetrics

# Stop script if no model is trained
if "topic_model" not in st.session_state.keys():
    st.error("Train a model to explore generated topics.", icon="🚨")
    st.stop()

if "topics_over_time"  not in st.session_state.keys():
    st.error("Topics over time required to plot topic map.", icon="🚨")
    st.stop()

### TITLE ###
st.title("Topics map")

st.slider("Time weight", min_value=0.0, max_value=0.1, value=0.04, key="tw", step=0.005)

### Main page ###
topic_metrics = TopicMetrics(st.session_state["topic_model"], st.session_state["topics_over_time"])
try:
	st.plotly_chart(topic_metrics.plot_TEM_map(st.session_state["tw"]))
except StatisticsError as se:
	st.warning(f"Try to change the Time Weight value: {se}", icon="⚠️")