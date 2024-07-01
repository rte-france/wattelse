#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import hmac
import numpy as np
import pandas as pd
import streamlit as st
import sqlite3

from pathlib import Path
from plotly.express import bar
import plotly.graph_objects as go

from wattelse.chatbot.frontend.django_chatbot.settings import DB_DIR

DB_PATH = DB_DIR / "db.sqlite3"
DATA_TABLE = "chatbot_chat"
USER_TABLE = "auth_user"

# Feedback identifiers in the database
GREAT = "great"
OK = "ok"
MISSING = "missing_info"
WRONG = "wrong"

# Color mapping for feedback values
FEEDBACK_COLORS = {
    GREAT: "green",
    OK: "blue",
    MISSING: "orange",
    WRONG: "red",
}


def get_db_data(path_to_db: Path) -> pd.DataFrame:
    con = sqlite3.connect(path_to_db)

    # Get column names from the table
    cur = con.cursor()
    cur.execute(
        f"SELECT username, group_id, conversation_id, message, response, answer_timestamp, answer_delay,"
        f"short_feedback, long_feedback "
        f"FROM {DATA_TABLE}, {USER_TABLE} "
        f"WHERE {DATA_TABLE}.user_id = {USER_TABLE}.id"
    )
    column_names = [
        desc[0] for desc in cur.description
    ]  # Get column names from description
    data = cur.fetchall()

    # Create DataFrame with column names
    df = pd.DataFrame(data, columns=column_names)
    df.answer_timestamp = pd.to_datetime(df.answer_timestamp)
    con.close()
    return df


def side_bar():
    ### SIDEBAR OPTIONS ###
    with st.sidebar.form("parameters_sidebar"):
        st.title("Parameters")

        st.selectbox(
            "Select user",
            st.session_state["full_data"].username.unique(),
            index=None,
            placeholder="Select user...",
            key="user",
        )
        st.selectbox(
            "Select group",
            st.session_state["full_data"].group_id.unique(),
            index=None,
            placeholder="Select group...",
            key="group",
        )

        # Select time range
        min_max = st.session_state["full_data"]["answer_timestamp"].agg(["min", "max"])
        if "timestamp_range" not in st.session_state:
            st.session_state["timestamp_range"] = (
                min_max["min"].to_pydatetime(),
                min_max["max"].to_pydatetime(),
            )
        st.slider(
            "Select the range of timestamps",
            min_value=min_max["min"].to_pydatetime(),
            max_value=min_max["max"].to_pydatetime(),
            key="timestamp_range",
        )

        parameters_sidebar_clicked = st.form_submit_button(
            "Update", type="primary", on_click=filter_data
        )
    return parameters_sidebar_clicked


def filter_data():
    filtered = st.session_state["full_data"]
    if st.session_state["user"]:
        filtered = filtered[filtered.username == st.session_state["user"]]
    if st.session_state["group"]:
        filtered = filtered[filtered.group_id == st.session_state["group"]]

    # Filter dataset to select only text within time range
    timestamp_range = st.session_state["timestamp_range"]
    filtered = filtered.query(
        f"answer_timestamp >= '{timestamp_range[0]}' and answer_timestamp <= '{timestamp_range[1]}'"
    )

    st.session_state["filtered_data"] = filtered


def display_indicators():
    nb_questions = len(st.session_state["filtered_data"].message)
    nb_conversations = len(st.session_state["filtered_data"].conversation_id.unique())
    avg_nb_questions = len(st.session_state["full_data"].message) // len(
        st.session_state["full_data"].username.unique()
    )
    avg_nb_conversations = len(
        st.session_state["full_data"].conversation_id.unique()
    ) // len(st.session_state["full_data"].username.unique())
    nb_short_feedback = (st.session_state["filtered_data"].short_feedback != "").sum()
    nb_long_feedback = (st.session_state["filtered_data"].long_feedback != "").sum()
    median_answer_delay = st.session_state["filtered_data"].answer_delay.median()/1e6

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(
        "Questions/Answers",
        nb_questions,
        (
            f"{(nb_questions - avg_nb_questions) / avg_nb_questions * 100:.1f}%"
            if st.session_state["user"]
            else ""
        ),
    )
    col2.metric(
        "Conversations",
        nb_conversations,
        (
            f"{(nb_conversations - avg_nb_conversations) / avg_nb_conversations * 100:.1f}%"
            if st.session_state["user"]
            else ""
        ),
    )
    if nb_conversations > 0:
        ratio = nb_questions / nb_conversations
        avg_ratio = avg_nb_questions / avg_nb_conversations
        col3.metric(
            "Questions per conversation",
            f"{ratio:.1f}",
            (
                f"{(ratio - avg_ratio) / avg_ratio * 100:.1f}%"
                if st.session_state["user"]
                else ""
            ),
        )

    col4.metric(
        "Long feedback",
        f"{nb_long_feedback}",
    )


    col5.metric(
        "Short feedback percentage",
        f"{nb_short_feedback/nb_questions*100:.2f}%",
    )

    col6.metric(
        "Median answer delay",
        f"{median_answer_delay:.2f}s",
    )


def display_questions_over_time():
    df = st.session_state["filtered_data"]

    # Resample data by day and count messages
    message_counts = df.resample("D", on="answer_timestamp")["message"].count()

    # Create plotly bar chart
    fig = bar(
        message_counts.to_frame("count"),
        x=message_counts.index,
        y="count",
        title="Number of Messages per Day",
    )

    # Customize the chart (optional)
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Messages")
    fig.update_traces(marker_color="lightblue")  # Change bar color

    # Display the chart in Streamlit
    st.plotly_chart(fig)


def display_feedback_charts():
    df = st.session_state["filtered_data"]

    # Filter data for feedback values in the color map
    filtered_df = df[df["short_feedback"].isin(FEEDBACK_COLORS.keys())]

    # Count occurrences of each short_feedback value in the filtered data
    short_feedback_counts = filtered_df["short_feedback"].value_counts().reindex(FEEDBACK_COLORS.keys())

    # Create a bar chart for total counts with custom colors
    fig_short_feedback_total = bar(
        short_feedback_counts,
        x=short_feedback_counts.index,
        y="count",
        title="Total Count of Short Feedback Values",
    )

    # Customize the chart layout and colors
    fig_short_feedback_total.update_layout(
        xaxis_title="Short Feedback", yaxis_title="Number of feedback"
    )
    fig_short_feedback_total.update_traces(
        marker_color=[FEEDBACK_COLORS[val] for val in short_feedback_counts.index]
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig_short_feedback_total)


def display_feedback_charts_over_time():
    df = st.session_state["filtered_data"]

    # Filter data for feedback values in the color map
    filtered_df = df[df["short_feedback"].isin(FEEDBACK_COLORS.keys())]

    # Resample data by day and count occurrences of each short_feedback value
    short_feedback_daily = (
        filtered_df.resample("D", on="answer_timestamp")["short_feedback"]
        .value_counts()
        .unstack()
    )

    # Get list of feedback values (column names)
    feedback_values = list(short_feedback_daily.columns)

    # Create traces for each feedback value with its corresponding color
    data = []
    for i, feedback in enumerate(feedback_values):
        data.append(
            go.Bar(
                x=short_feedback_daily.index,
                y=short_feedback_daily[feedback],
                name=feedback,
                marker=dict(color=FEEDBACK_COLORS[feedback]),
            )
        )

    # Create layout with title and legend
    layout = go.Layout(
        title="Daily Short Feedback Counts", xaxis_title="Date", yaxis_title="Count"
    )

    # Display chart on Streamlit
    st.plotly_chart(go.Figure(data=data, layout=layout).update_layout(barmode="stack"))


def display_feedback_rates():
    filtered_df = st.session_state["filtered_data"]
    short_feedback_counts = filtered_df["short_feedback"].value_counts()
    total_short_feedback = (st.session_state["filtered_data"].short_feedback != "").sum()
    cols = st.columns(4)
    for i, feedback_type in enumerate(FEEDBACK_COLORS.keys()):
        if feedback_type in short_feedback_counts.keys():
            cols[i].metric(
                    f":{FEEDBACK_COLORS[feedback_type]}[Ratio of feedback '{feedback_type}']",
                    f"{short_feedback_counts[feedback_type]/total_short_feedback*100:.1f}%",
                    "",
                )
            
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False
            


def main():
    # Wide layout
    st.set_page_config(page_title="Wattelse dashboard", layout="wide")

    # Title
    st.title("Wattelse dashboard")

    # Password
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    # Load data
    st.session_state["full_data"] = get_db_data(DB_PATH)

    if side_bar():
        with st.expander("Raw data"):
            st.write(st.session_state["filtered_data"])

        # High level indicators per user / group depending on the selection
        with st.expander("High level indicators", expanded=True):
            display_indicators()

        with st.expander("Questions over time", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                display_questions_over_time()
            with col2:
                # empty placeholder
                pass

        with st.expander("Feedback rates", expanded=True):
            display_feedback_rates()

        with st.expander("Short feedback", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                display_feedback_charts()
            with col2:
                display_feedback_charts_over_time()


main()
