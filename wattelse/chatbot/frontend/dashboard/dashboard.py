#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import yaml
import pandas as pd
import streamlit as st

from pathlib import Path

from pandas.core.dtypes.cast import maybe_infer_to_datetimelike

from wattelse.chatbot.frontend.dashboard.dashboard_utils import (
    DRH_GROUP_NAME, 
    METIERS_GROUP_NAME,
    DATA_TABLES,
    get_db_data,
    filter_data,
    check_password,
    _compute_file_indicators,
    build_msg_df_over_time,
    build_users_df,
    build_users_satisfaction_over_nb_eval,
)
from wattelse.chatbot.frontend.dashboard.dashboard_display import (
    display_feedback_charts,
    display_feedback_charts_over_time,
    display_feedback_rates,
    display_indicators,
    display_user_graph,
    display_user_hist_over_eval,
    display_users_satisfaction_over_nb_eval
)
from wattelse.chatbot.frontend.django_chatbot.settings import DB_DIR

DB_PATH = DB_DIR / "db.sqlite3"

# Get Expé_Métiers group names list
GROUP_NAMES_LIST_FILE_PATH = Path(__file__).parent / "expe_metier_group_name_list.yaml"
with open(GROUP_NAMES_LIST_FILE_PATH) as f:
    GROUP_NAMES_LIST = yaml.safe_load(f)


def side_bar():
    ### SIDEBAR OPTIONS ###
    with st.sidebar.form("parameters_sidebar"):
        st.title("Parameters")

        # Get user and group names and sort them
        user_names_list = list(st.session_state["full_data"].username.unique())
        user_names_list.sort(key=str.lower)
        group_names_list = list(st.session_state["full_data"].group_id.unique())
        group_names_list.sort(key=str.lower)

        # Format group_names_list so DRH and Expé_Métiers are at the top of the list
        # Only if DRH_GROUP_NAME in group_names (so this option only appear on server 1)
        if DRH_GROUP_NAME in group_names_list:
            group_names_list.remove(DRH_GROUP_NAME)
            group_names_list.insert(0, DRH_GROUP_NAME)
            group_names_list.insert(0, METIERS_GROUP_NAME)

        st.selectbox(
            "Select user",
            user_names_list,
            index=None,
            placeholder="Select user...",
            key="user",
        )
        st.selectbox(
            "Select group",
            group_names_list,
            index=None,
            placeholder="Select group...",
            key="group",
        )

        # Select time range
        min_max = st.session_state["full_data"]["answer_timestamp"].agg(["min", "max"])
        min_date = min_max["min"].to_pydatetime()
        max_date = min_max["max"].to_pydatetime()
        if "timestamp_range" not in st.session_state:
            st.session_state["timestamp_range"] = (
                min_date,
                max_date,
            )
        st.slider(
            "Select the range of timestamps",
            min_value=min_date,
            max_value=(
                max_date if min_date != max_date else min_date + pd.Timedelta(minutes=1)
            ),  # to avoid slider errors
            key="timestamp_range",
        )

        parameters_sidebar_clicked = st.form_submit_button(
            "Update", type="primary", on_click=filter_data
        )
    return parameters_sidebar_clicked




def main():
    if "selected_table" not in st.session_state:
        st.session_state["selected_table"] = list(DATA_TABLES)[0]

    # Wide layout
    st.set_page_config(
        page_title=f"Wattelse dashboard for {st.session_state['selected_table']}",
        layout="wide",
    )

    # Title
    st.title(f"Wattelse dashboard for {st.session_state['selected_table']}")

    # Password
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    # Select data
    st.selectbox(
        "Data: RAG or secureGPT?",
        list(DATA_TABLES.keys()),
        placeholder="Select data table (RAG or secureGPT)",
        key="selected_table",
    )

    # Load data
    st.session_state["full_data"] = get_db_data(DB_PATH)

    if side_bar():
        filtered_df = st.session_state["filtered_data"]

        with st.expander("Raw data"):
            st.write(filtered_df.sort_values(by="answer_timestamp", ascending=False))

        # High level indicators per user / group depending on the selection
        with st.expander("High level indicators", expanded=True):
            number_of_files, number_of_chunks = _compute_file_indicators(group=st.session_state["group"])

            display_indicators(
                filtered_df=filtered_df, 
                number_of_files=number_of_files, 
                number_of_chunks=number_of_chunks
                )

        with st.expander("Feedback rates", expanded=True):
            display_feedback_rates(filtered_df=filtered_df)

        with st.expander("Short feedback", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                display_feedback_charts(filtered_df=filtered_df)
            with col2:
                pass
            msg_df = build_msg_df_over_time(filtered_df=filtered_df)
            display_feedback_charts_over_time(msg_df=msg_df)

        with st.expander("Users analysis", expanded=True):
            users_df = build_users_df(filtered_df=filtered_df)
            display_user_graph(users_df=users_df)
            users_satisfaction = build_users_satisfaction_over_nb_eval(
                users_df=users_df
            )
            display_users_satisfaction_over_nb_eval(
                users_satisfaction=users_satisfaction
            )
            display_user_hist_over_eval(users_df=users_df)

        with st.expander("Users raw data", expanded=False):
            st.write(users_df)



main()
