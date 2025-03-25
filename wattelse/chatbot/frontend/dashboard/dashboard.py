#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import yaml
from datetime import timedelta
import streamlit as st

from pathlib import Path

from wattelse.chatbot.frontend.dashboard.indicators import (
    _compute_file_indicators,
    build_msg_df_over_time,
    build_users_df,
    build_users_satisfaction_over_nb_eval,
    build_extracts_df,
    build_extracts_pivot,
)

from wattelse.chatbot.frontend.dashboard.dashboard_utils import (
    DRH_GROUP_NAME,
    METIERS_GROUP_NAME,
    DATA_TABLES,
    initialize_state_session,
    update_state_session,
    check_password,
)

from wattelse.chatbot.frontend.dashboard.dashboard_display import (
    display_feedback_charts,
    display_feedback_charts_over_time,
    display_feedback_rates,
    display_indicators,
    display_unique_visitors,
    display_user_graph,
    display_user_hist_over_eval,
    display_users_satisfaction_over_nb_eval,
    display_extracts_graph,
)


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
        user_names_list = list(st.session_state["unfiltered_data"].username.unique())
        user_names_list.sort(key=str.lower)
        group_names_list = list(st.session_state["unfiltered_data"].group_id.unique())
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
            min_value=1,
            max_value=100,
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
        full_df = st.session_state["unfiltered_data"]
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
                full_df=full_df,
                number_of_files=number_of_files,
                number_of_chunks=number_of_chunks,
            )
        with st.expander("Weekly unique users"):
            st.plotly_chart(display_unique_visitors(filtered_df))

        with st.expander("Feedback rates", expanded=True):
            display_feedback_rates(filtered_df=filtered_df)

        with st.expander("Short feedback", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                fig = display_feedback_charts(filtered_df=filtered_df)
                st.plotly_chart(fig)
            with col2:
                pass
            msg_df = build_msg_df_over_time(
                filtered_df=filtered_df, nb_reponse_lissage=nb_reponse_lissage
            )
            fig = display_feedback_charts_over_time(
                msg_df=msg_df, nb_reponse_lissage=nb_reponse_lissage
            )
            st.plotly_chart(fig)

        with st.expander("Users analysis", expanded=True):
            users_df = build_users_df(filtered_df=filtered_df)
            if len(users_df) > 0:
                fig = display_user_graph(users_df=users_df)
                st.plotly_chart(fig)
                users_satisfaction = build_users_satisfaction_over_nb_eval(
                    users_df=users_df
                )
                fig = display_users_satisfaction_over_nb_eval(
                    users_satisfaction=users_satisfaction
                )
                st.plotly_chart(fig)
                fig = display_user_hist_over_eval(users_df=users_df)
                st.plotly_chart(fig)
            else:
                st.write("no user to display")

        with st.expander("Users raw data", expanded=False):
            st.write(users_df)

        if st.session_state["selected_table"] == "RAG":
            with st.expander("Relevant extracts analysis", expanded=True):
                relevant_extracts_df = build_extracts_df(filtered_df=filtered_df)
                if len(relevant_extracts_df) > 0:
                    extracts_pivot = build_extracts_pivot(
                        extracts_pivot=relevant_extracts_df
                    )

                    fig = display_extracts_graph(extracts_pivot=extracts_pivot)
                    st.plotly_chart(fig)
                else:
                    st.write("no relevant extract to display")

            with st.expander(f"Filtrage des extraits", expanded=False):

                if len(relevant_extracts_df) > 0:
                    filtered_extracts_df = relevant_extracts_df.loc[
                        relevant_extracts_df["content"].str.contains(
                            st.session_state["filter_str"]
                        )
                    ]

                    st.write(
                        f"{filtered_extracts_df.shape[0]} réponses ont utilisées cet extrait"
                    )
                    if len(filtered_extracts_df) > 0:
                        st.dataframe(data=filtered_extracts_df)


main()
