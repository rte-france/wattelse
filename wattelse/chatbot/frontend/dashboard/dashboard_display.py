#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import pandas as pd
import streamlit as st


from plotly.express import bar
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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



def display_indicators(filtered_df: pd.DataFrame, number_of_files:int, number_of_chunks:int):
    nb_questions = len(filtered_df.message)
    nb_conversations = len(filtered_df.conversation_id.unique())
    avg_nb_questions = len(st.session_state["full_data"].message) // len(
        st.session_state["full_data"].username.unique()
    )
    avg_nb_conversations = len(
        st.session_state["full_data"].conversation_id.unique()
    ) // len(st.session_state["full_data"].username.unique())
    nb_short_feedback = (filtered_df.short_feedback != "").sum()
    nb_long_feedback = (filtered_df.long_feedback != "").sum()
    median_answer_delay = filtered_df.answer_delay.median() / 1e6


    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
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
        f"{nb_short_feedback / nb_questions * 100:.2f}%",
    )

    col6.metric(
        "Median answer delay",
        f"{median_answer_delay:.2f}s",
    )

    col7.metric("Number of files", f"{number_of_files}")

    col8.metric("Number of chunks", f"{number_of_chunks}")




def display_feedback_charts_over_time(msg_df: pd.DataFrame):

    # Création de l'histogramme empilé
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Ajout de l'histogramme empilé
    fig.add_trace(
        go.Bar(
            x=msg_df.index,
            y=msg_df["wrong"],
            name="réponse fausse",
            marker_color="red",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=msg_df.index,
            y=msg_df["missing_info"],
            name="réponse incomplète",
            marker_color="orange",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=msg_df.index,
            y=msg_df["ok"],
            name="réponse correcte",
            marker_color="blue",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=msg_df.index,
            y=msg_df["great"],
            name="réponse excellente",
            marker_color="green",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=msg_df.index,
            y=msg_df["non_evalue"],
            name="pas de réponse",
            marker_color="grey",
        ),
        secondary_y=False,
    )

    # Ajout des courbes
    fig.add_trace(
        go.Scatter(
            x=msg_df.index,
            y=msg_df["nb_feedback_long"],
            mode="markers",
            name="nombre de réponses longues",
            line=dict(color="coral"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=msg_df.index,
            y=msg_df["tx_feedback"],
            mode="markers",
            name="%age de réponses évaluées",
            line=dict(color="purple"),
        ),
        secondary_y=True,
    )

    # Mise à jour de la mise en page
    fig.update_layout(
        title="Activity over time",
        xaxis_title="Date",
        yaxis_title="Number",
        yaxis2_title="Percentage",
        barmode="stack",
    )
    st.plotly_chart(fig)

    return


def display_feedback_charts(filtered_df: pd.DataFrame):

    # Filter data for feedback values in the color map
    filtered_df = filtered_df[
        filtered_df["short_feedback"].isin(FEEDBACK_COLORS.keys())
    ]

    # Count occurrences of each short_feedback value in the filtered data
    short_feedback_counts = (
        filtered_df["short_feedback"].value_counts().reindex(FEEDBACK_COLORS.keys())
    )

    # Create a bar chart for total counts with custom colors
    fig_short_feedback_total = bar(
        short_feedback_counts,
        x=short_feedback_counts.index,
        y="count",
        title="Total count of evaluated answers",
    )

    # Customize the chart layout and colors
    fig_short_feedback_total.update_layout(
        xaxis_title="Evaluations", yaxis_title="Number of feedback"
    )
    fig_short_feedback_total.update_traces(
        marker_color=[FEEDBACK_COLORS[val] for val in short_feedback_counts.index]
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig_short_feedback_total)


def display_feedback_rates(filtered_df: pd.DataFrame):

    short_feedback_counts = filtered_df["short_feedback"].value_counts()
    total_short_feedback = (
        st.session_state["filtered_data"].short_feedback != ""
    ).sum()
    cols = st.columns(4)
    for i, feedback_type in enumerate(FEEDBACK_COLORS.keys()):
        if feedback_type in short_feedback_counts.keys():
            cols[i].metric(
                f":{FEEDBACK_COLORS[feedback_type]}[Ratio of feedback '{feedback_type}']",
                f"{short_feedback_counts[feedback_type] / total_short_feedback * 100:.1f}%",
                "",
            )


def display_user_graph(users_df: pd.DataFrame) -> None:
    """Generate a plotly graph showing the number of questions and evaluation of each users

    Args:
        users_df (pd.DataFrame): users_df, as build by the above function build_users_df
    """
    # Création de l'histogramme empilé
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Ajout de l'histogramme empilé
    fig.add_trace(
        go.Bar(
            x=users_df.index,
            y=users_df["wrong"],
            name="réponse fausse",
            marker_color="red",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=users_df.index,
            y=users_df["missing_info"],
            name="réponse incomplète",
            marker_color="orange",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=users_df.index,
            y=users_df["ok"],
            name="réponse correcte",
            marker_color="blue",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=users_df.index,
            y=users_df["great"],
            name="réponse excellente",
            marker_color="green",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=users_df.index,
            y=users_df["non_evalue"],
            name="pas de réponse",
            marker_color="grey",
        ),
        secondary_y=False,
    )
    # Ajout des courbes
    fig.add_trace(
        go.Scatter(
            x=users_df.index,
            y=users_df["nb_feedback_long"],
            mode="lines+markers",
            name="nombre de réponses longues",
            line=dict(color="coral"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=users_df.index,
            y=users_df["tx_feedback"],
            mode="markers",
            name="%age de réponses évaluées",
            line=dict(color="purple"),
        ),
        secondary_y=True,
    )

    # Mise à jour de la mise en page
    fig.update_layout(
        title="Retours par utilisateur",
        xaxis_title="Utilisateurs",
        yaxis_title="Nombre",
        yaxis2_title="Pourcentage",
        barmode="stack",
    )
    st.plotly_chart(fig)
    return


def display_users_satisfaction_over_nb_eval(
    users_satisfaction: pd.DataFrame,
) -> pd.DataFrame:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=[
                1.0 / 4,
                1.0 / 4,
            ],
            x=[users_satisfaction.index[0], users_satisfaction.index[-1]],
            mode="none",
            name="mauvais",
            # "red"
            fillcolor="rgba(255, 0, 0, 0.2)",
            stackgroup="one",  # define stack group
        )
    )
    fig.add_trace(
        go.Scatter(
            y=[
                1.0 / 4,
                1.0 / 4,
            ],
            x=[users_satisfaction.index[0], users_satisfaction.index[-1]],
            mode="none",
            name="info manquante",
            # "orange"
            fillcolor="rgba(255, 128, 0, 0.2)",
            stackgroup="one",  # define stack group
        )
    )
    fig.add_trace(
        go.Scatter(
            y=[
                1.0 / 4,
                1.0 / 4,
            ],
            x=[users_satisfaction.index[0], users_satisfaction.index[-1]],
            mode="none",
            name="ok",
            # "green"
            fillcolor="rgba(0, 255, 0, 0.2)",
            stackgroup="one",  # define stack group
        )
    )
    fig.add_trace(
        go.Scatter(
            y=[
                1.0 / 4,
                1.0 / 4,
            ],
            x=[users_satisfaction.index[0], users_satisfaction.index[-1]],
            mode="none",
            name="excellent",
            # "blue"
            fillcolor="rgba(0, 0, 255, 0.2)",
            stackgroup="one",  # define stack group
        )
    )
    fig.add_trace(
        go.Bar(
            y=users_satisfaction["eval_mean"],
            x=users_satisfaction.index,
            name="eval moyenne",
            marker_color="blue",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=users_satisfaction["eval_std"],
            x=users_satisfaction.index,
            mode="lines",
            name="variabilité moyenne par utilisateurs",
            line=dict(color="lightblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=users_satisfaction["eval_mean_std"],
            x=users_satisfaction.index,
            mode="lines",
            name="variabilité moyenne entre utilisateurs",
            line=dict(color="lightgreen"),
        )
    )
    fig.update_layout(
        title="Evaluation moyenne des réponses par utilisateurs, en fonction du nombre d'évaluations réalisées",
        xaxis_title="nb d'évaluations réalisées",
        yaxis_title="Evaluation moyenne, entre 0 et 1",
    )
    st.plotly_chart(fig)
    return


def display_user_hist_over_eval(users_df):

    distinct_count_with_na = users_df["evalue"].value_counts()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=distinct_count_with_na.index, y=distinct_count_with_na.values)
    )
    fig.update_layout(
        title="Histogramme des utilisateurs par nb réponses évaluées",
        yaxis_title="nombre d'utilisateur",
        xaxis_title="nombre de questions évaluées",
    )
    fig.update_layout(
        title="Histogramme des utilisateurs, en fonction du nombre d'évaluations réalisées",
        yaxis_title="nombre d'utilisateurs",
        xaxis_title="nombre de questions évaluées",
    )
    st.plotly_chart(fig)
    return
