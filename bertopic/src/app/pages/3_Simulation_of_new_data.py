from collections import Counter
from typing import List, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from metrics import TopicMetrics, TIME_WEIGHT, TEM_x, TEM_y
from pandas import DataFrame
from state_utils import register_widget, save_widget_state, restore_widget_state
from utils import TIMESTAMP_COLUMN, TEXT_COLUMN

# Restore widget state
restore_widget_state()

if "tw" not in st.session_state.keys():
    st.session_state["tw"] = TIME_WEIGHT


def main():
    # Stop script if no model is trained
    if "topic_model" not in st.session_state.keys():
        st.error("Train a model to explore the impact of new data on topics.", icon="🚨")
        st.stop()

    if "topics_over_time" not in st.session_state.keys():
        st.error("Topics over time required.", icon="🚨")
        st.stop()


    ### TITLE ###
    st.title("Simulation of topic evolution with new data")

    timestamp_max = st.session_state["timefiltered_df"][TIMESTAMP_COLUMN].max()
    st.session_state["remaining_df"] = st.session_state["raw_df"].query(
        f"timestamp > '{timestamp_max}'"
    )

    # Handle missing data
    if st.session_state["remaining_df"].empty:
        st.error("Not enough remaining data to simulate evolution of topics", icon="🚨")
        st.info(
            "Please reduce the max value of the dataset timestamp in the main page!",
            icon="ℹ️",
        )
        st.stop()

    # Display remaining data
    st.write(f"Remaining data: {len(st.session_state['remaining_df'])} documents.")
    with st.expander("View data"):
        st.dataframe(st.session_state["remaining_df"].head())

    # Selection of number of batches
    register_widget("new_data_batches_nb")
    st.slider(
        "Number of data batches",
        min_value=1,
        max_value=min(10, len(st.session_state["remaining_df"])),
        key="new_data_batches_nb",
        on_change=save_widget_state
    )

    register_widget("tw")
    st.slider("Time weight", min_value=0.0, max_value=0.1, step=0.005, key="tw", on_change=save_widget_state)

    if st.button("Run", type="primary"):
        # computes predictions and related data per batch
        batch_results = process_new_data_per_batch()

        # use this data for various display
        # - plot animated topic map
        plot_animated_topic_map(batch_results)
        # - TODO: other visualizations?

def process_new_data_per_batch() -> List[Dict]:
    """Enriches topics with new data, returns a list (each element of the list correspond to one batch and is  """
    df_batches = np.array_split(
        st.session_state["remaining_df"], st.session_state["new_data_batches_nb"]
    )

    new_topics_over_time = st.session_state["topics_over_time"]

    results = []

    # Need to sort batches by ascending time for correct processing order
    ts_avg = {}
    for i, df_b in enumerate(df_batches):
        ts_avg[i] = df_b[TIMESTAMP_COLUMN].mean()

    # Process set of batches
    for i, ts_avg in sorted(ts_avg.items()):
        batch_id = i+1
        with st.spinner(f"Simulating batch #{batch_id} as new incoming data..."):

            # For each batch, computes the topic map and combines the result into a common dataframe

            # classify new items
            topics, probs = st.session_state["topic_model"].transform(list(df_batches[i][TEXT_COLUMN]))

            # update model stats
            topics_counter = Counter(topics)
            words = "" # TODO: update words in the same way as it is done in bertopic (kind of tf-idf)
            topics_over_time_per_batch = [
                {"Topic": t, "Frequency": f, "Words": words, "Timestamp": ts_avg, "Batch": batch_id}
                for t, f in topics_counter.items()
            ]

            # Combine the batch result with existing info about topics over time
            new_topics_over_time = pd.concat([new_topics_over_time, pd.DataFrame(topics_over_time_per_batch)])

            results.append({"ts_avg": ts_avg, "topics": topics, "probs": probs, "topics_over_time": new_topics_over_time})

    return results

def plot_animated_topic_map(batch_results: List[Dict]):
    # Topic map based on data before the introduction of new batches
    tm = TopicMetrics(st.session_state["topic_model"], st.session_state["topics_over_time"])
    TEM_map = tm.TEM_map(st.session_state["tw"])
    TEM_map = tm.identify_signals(TEM_map, TEM_x, TEM_y)
    TEM_map["batch"] = 0

    # Use batch results to compute new data
    for i, res in enumerate(batch_results):
        # New topic metrics (that takes into account the new batch)
        topic_metrics = TopicMetrics(st.session_state["topic_model"], res["topics_over_time"])
        batch_TEM_map = topic_metrics.TEM_map(st.session_state["tw"])
        batch_TEM_map = topic_metrics.identify_signals(batch_TEM_map, TEM_x, TEM_y)
        batch_TEM_map["batch"] = i+1
        TEM_map = pd.concat([TEM_map, batch_TEM_map])


    # Plot the resulting map as an animation
    with st.spinner("Plotting topic map..."):
        st.plotly_chart(TopicMetrics.scatterplot_with_annotations(TEM_map, TEM_x, TEM_y, "topic", "topic_description",
                                                                  "Animated Topic Emergence Map (TEM)", TEM_x, TEM_y,
                                                                  animation_frame="batch"
                                                                  ))

###
# Write page
main()