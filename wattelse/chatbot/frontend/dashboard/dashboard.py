#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import yaml
import pandas as pd
from datetime import timedelta
import streamlit as st

from pathlib import Path

from pandas.core.dtypes.cast import maybe_infer_to_datetimelike

from wattelse.chatbot.frontend.dashboard.dashboard_utils import (
    DRH_GROUP_NAME,
    METIERS_GROUP_NAME,
    DATA_TABLES,
    initialize_state_session,
    update_state_session,
    check_password,
    _compute_file_indicators,
    build_msg_df_over_time,
    build_users_df,
    build_users_satisfaction_over_nb_eval,
    build_extracts_df,
    build_extracts_pivot,
)
from wattelse.chatbot.frontend.dashboard.dashboard_display import (
    display_feedback_charts,
    display_feedback_charts_over_time,
    display_feedback_rates,
    display_indicators,
    display_user_graph,
    display_user_hist_over_eval,
    display_users_satisfaction_over_nb_eval,
    display_extracts_graph,
)


# Get Expé_Métiers group names list
GROUP_NAMES_LIST_FILE_PATH = Path(__file__).parent / "expe_metier_group_name_list.yaml"
with open(GROUP_NAMES_LIST_FILE_PATH) as f:
    GROUP_NAMES_LIST = yaml.safe_load(f)


def side_bar():
    ### SIDEBAR OPTIONS ###
    with st.sidebar.form("parameters_sidebar"):
        st.title("Parameters")

        # Select data
        st.selectbox(
            "Data: RAG or secureGPT?",
            list(DATA_TABLES.keys()),
            placeholder="Select data table (RAG or secureGPT)",
            key="selected_table",
        )

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
        (min_date, max_date) = st.session_state["unfiltered_timestamp_range"]
        st.slider(
            "Select the range of timestamps",
            min_value=min_date,
            max_value=max(
                max_date, min_date + timedelta(days=1)
            ),  # to avoid slider errors
            value=(min_date, max_date),
            step=timedelta(days=1),
            key="timestamp_range",
        )

        st.slider(
            "Select the number of feedback to smooth over",
            min_value = 1,
            max_value = 100,
            value=15,
            step=1,
            key="nb_reponse_lissage",
        )

        if st.session_state["selected_table"] == "RAG":
            st.text_input(
                label="Saisir ici un extrait de document à retrouver",
                value=st.session_state["extract_substring"],
                max_chars=None,
                type="default",
                help="Saisir ici un extrait de document à retrouver.",
                key="filter_str",
            )

            st.session_state["extract_substring"] = st.session_state["filter_str"]

        parameters_sidebar_clicked = st.form_submit_button(
            "Update", type="primary", on_click=update_state_session
        )

    return parameters_sidebar_clicked


def main():

    initialize_state_session()

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

    if side_bar():
        filtered_df = st.session_state["filtered_data"]
        nb_reponse_lissage = st.session_state["nb_reponse_lissage"]

        with st.expander("Raw data"):
            st.write(filtered_df.sort_values(by="answer_timestamp", ascending=False))

        # High level indicators per user / group depending on the selection
        with st.expander("High level indicators", expanded=True):
            number_of_files, number_of_chunks = _compute_file_indicators(
                group=st.session_state["group"]
            )

            display_indicators(
                filtered_df=filtered_df,
                number_of_files=number_of_files,
                number_of_chunks=number_of_chunks,
            )

        with st.expander("Feedback rates", expanded=True):
            display_feedback_rates(filtered_df=filtered_df)

        with st.expander("Short feedback", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                display_feedback_charts(filtered_df=filtered_df)
            with col2:
                pass
            msg_df = build_msg_df_over_time(filtered_df=filtered_df, nb_reponse_lissage=nb_reponse_lissage)
            display_feedback_charts_over_time(msg_df=msg_df, nb_reponse_lissage=nb_reponse_lissage)

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

        if st.session_state["selected_table"] == "RAG":
            with st.expander("Relevant extracts analysis", expanded=True):
                relevant_extracts_df = build_extracts_df(filtered_df=filtered_df)

                extracts_pivot = build_extracts_pivot(
                    extracts_pivot=relevant_extracts_df
                )

                display_extracts_graph(extracts_pivot=extracts_pivot)

            with st.expander(f"Filtrage des extraits", expanded=False):

                filtered_extracts_df = relevant_extracts_df.loc[
                    relevant_extracts_df["content"].str.contains(
                        st.session_state["filter_str"]
                    )
                ]

                st.write(
                    f"{filtered_extracts_df.shape[0]} réponses ont utilisées cet extrait"
                )
                if filtered_extracts_df.shape[0] > 0:
                    st.dataframe(data=filtered_extracts_df)


main()
