import itertools
from typing import List

import pandas as pd
import streamlit as st
from pathlib import Path

from wattelse.bertopic.app.app_utils import plot_docs_reparition_over_time
from wattelse.bertopic.app.state_utils import register_widget, save_widget_state


def data_overview(df: pd.DataFrame):
    with st.expander("Data overview"):
        freq = st.select_slider(
            "Time aggregation",
            options=(
                "1D",
                "2D",
                "1W",
                "2W",
                "1M",
                "2M",
                "1Y",
                "2Y",
            ),
            value="1M",
        )
        plot_docs_reparition_over_time(df, freq)


def choose_data(base_dir: Path, filters: List[str]):
    data_folders = sorted(
        set(
            f.parent
            for f in itertools.chain.from_iterable(
                [list(base_dir.glob(f"**/{filter}")) for filter in filters]
            )
        )
    )
    if "data_folder" not in st.session_state:
        st.session_state["data_folder"] = data_folders[0]

    data_options = ["None"] + sorted(
        [
            p.name
            for p in itertools.chain.from_iterable(
                [
                    list(st.session_state["data_folder"].glob(f"{filter}"))
                    for filter in filters
                ]
            )
        ]
    )

    if "data_name" not in st.session_state:
        st.session_state["data_name"] = data_options[0]

    register_widget("data_name")
    register_widget("data_folder")
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.selectbox(
            "Base folder", data_folders, key="data_folder", on_change=reset_data
        )
    with col2:
        st.selectbox(
            "Select data to continue",
            data_options,
            key="data_name",
            on_change=reset_all,
        )

    # Stop the app as long as no data is selected
    if st.session_state["data_name"] == "None":
        st.stop()


def reset_all():
    # TODO: add here all state variables we want to reset when we change the data
    st.session_state.pop("timestamp_range", None)
    reset_topics()


def reset_data():
    # TODO: add here all state variables we want to reset when we change the data
    st.session_state.pop("timestamp_range", None)
    st.session_state.pop("data_name", None)
    reset_topics()


def reset_topics():
    # shall be called when we update data parameters (timestamp, min char, split, etc.)
    st.session_state.pop("selected_topic_number", None)
    st.session_state.pop("new_topics", None)
    st.session_state.pop("new_topics_over_time", None)
    save_widget_state()
