#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import hmac
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import sqlite3

from pathlib import Path

from pandas.core.dtypes.cast import maybe_infer_to_datetimelike
from plotly.express import bar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit import session_state

from wattelse.chatbot.backend.rag_backend import RAGBackEnd
from wattelse.chatbot.frontend.django_chatbot.settings import DB_DIR

DB_PATH = DB_DIR / "db.sqlite3"
DATA_TABLE_RAG = "chatbot_chat"
DATA_TABLE_GPT = "chatbot_gptchat"
USER_TABLE = "auth_user"

DATA_TABLES = {"RAG": DATA_TABLE_RAG, "SecureGPT": DATA_TABLE_GPT}

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

# Get main experimentations group names
DRH_GROUP_NAME = "DRH"
METIERS_GROUP_NAME = "Expé_Métiers"

# Get Expé_Métiers group names list
GROUP_NAMES_LIST_FILE_PATH = Path(__file__).parent / "expe_metier_group_name_list.yaml"
with open(GROUP_NAMES_LIST_FILE_PATH) as f:
    GROUP_NAMES_LIST = yaml.safe_load(f)


def get_db_data(path_to_db: Path) -> pd.DataFrame:
    """extract the django db of questions, answers and relevant_extracts. All is put in a dataframe

    Args:
        path_to_db (Path): the path where the db can be found

    Returns:
        pd.DataFrame: the resulting dataframe
    """    
    con = sqlite3.connect(path_to_db)

    # Get column names from the table
    cur = con.cursor()
    table = DATA_TABLES[st.session_state["selected_table"]]

    query = f"SELECT username, group_id, conversation_id, message, response, answer_timestamp, answer_delay, short_feedback, long_feedback"

    if st.session_state["selected_table"] == "RAG":
        query += ", relevant_extracts, group_system_prompt"
    query += f" FROM {table}, {USER_TABLE} WHERE {table}.user_id = {USER_TABLE}.id"

    cur.execute(query)
    column_names = [
        desc[0] for desc in cur.description
    ]  # Get column names from description
    data = cur.fetchall()

    # Create DataFrame with column names
    df = pd.DataFrame(data, columns=column_names)
    df.answer_timestamp = pd.to_datetime(df.answer_timestamp)
    con.close()

    df["long_feedback_bool"] = (df["long_feedback"] != "").astype(int)

    return df


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


def filter_data():
    """get the history dataframe from st.session_state["full_data"], 
    filter it with the values selected in the side bar (left part of the screen),
    write the result in st.session_state["filtered_data"]
    """    
    filtered = st.session_state["full_data"]
    if st.session_state["user"]:
        filtered = filtered[filtered.username == st.session_state["user"]]
    if st.session_state["group"]:
        if st.session_state["group"] == METIERS_GROUP_NAME:
            filtered = filtered[filtered.group_id.isin(GROUP_NAMES_LIST)]
        else:
            filtered = filtered[filtered.group_id == st.session_state["group"]]

    # Filter dataset to select only text within time range
    timestamp_range = st.session_state["timestamp_range"]
    filtered = filtered.query(
        f"answer_timestamp >= '{timestamp_range[0]}' and answer_timestamp <= '{timestamp_range[1]}'"
    )

    st.session_state["filtered_data"] = filtered

    return


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


def _compute_file_indicators():
    """Compute the number of files and number of chunks in the RAG backend that belong to the given group.

    Returns:
        _type_: _description_
    """    
    if st.session_state["group"] and st.session_state["group"] != METIERS_GROUP_NAME:
        bak = RAGBackEnd(st.session_state["group"])
        nb_files = len(bak.get_available_docs())
        nb_chunks = len(bak.document_collection.collection.get()["documents"])
    else:
        nb_files = np.NaN
        nb_chunks = np.NaN

    return nb_files, nb_chunks


def display_indicators(filtered_df: pd.DataFrame):
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

    number_of_files, number_of_chunks = _compute_file_indicators()

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


def build_msg_df_over_time(filtered_df: pd.DataFrame):

    message_daily = filtered_df.resample("D", on="answer_timestamp").agg(
        {"message": "count", "long_feedback_bool": "sum"}
    )

    # Resample data by day and count occurrences of each short_feedback value
    short_feedback_daily = (
        filtered_df.resample("D", on="answer_timestamp")["short_feedback"]
        .value_counts()
        .unstack()
    )

    message_daily = short_feedback_daily.join(message_daily)
    message_daily.fillna(value=0, inplace=True)
    message_daily.rename(
        columns={
            "message": "nb_questions",
            "": "non_evalue",
            "long_feedback_bool": "nb_feedback_long",
        },
        inplace=True,
    )
    for feedback in ["great", "ok", "missing_info", "wrong"]:
        if feedback not in message_daily.columns:
            message_daily.feedback = 0
    message_daily["tx_feedback"] = (
        1 - message_daily["non_evalue"] / message_daily["nb_questions"]
    )

    return message_daily


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
        title="Questions par jour",
        xaxis_title="Date",
        yaxis_title="Nombre",
        yaxis2_title="Pourcentage",
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


def build_users_df(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Given the log of each question, and evaluation, build a pivot table by user :
        - nb of question
        - nb of evaluation "great"
        - nb of evaluation "ok"
        - nb of evaluation "missing_info"
        - nb of evaluation "wrong"
        - nb of long feedback
        - proportion of answers evaluated

    Args:
        filtered_df (pd.DataFrame): raw log data

    Returns:
        pd.DataFrame: the pivot table described above.
    """
    users_feedback = pd.pivot_table(
        data=filtered_df[["username", "short_feedback", "answer_timestamp"]],
        index="username",
        values="answer_timestamp",
        columns="short_feedback",
        aggfunc="count",
    )
    data = filtered_df[
        ["username", "group_id", "conversation_id", "response", "long_feedback"]
    ]
    data["long_feedback_bool"] = (data["long_feedback"] != "").astype(int)

    users_df = data.groupby(
        by="username",
    ).agg(
        {
            "conversation_id": lambda x: len(x.unique()),
            "response": "count",
            "long_feedback_bool": "sum",
        }
    )
    users_df = users_df.join(users_feedback)
    users_df.fillna(value=0, inplace=True)
    users_df.sort_values(by="response", ascending=False, inplace=True)
    users_df.rename(
        columns={
            "conversation_id": "nb_conversation",
            "response": "nb_questions",
            "long_feedback_bool": "nb_feedback_long",
            "": "non_evalue",
        },
        inplace=True,
    )

    for feedback in ["great", "ok", "missing_info", "wrong"]:
        if feedback not in users_df.columns:
            users_df.feedback = 0

    users_df["tx_feedback"] = 1 - users_df["non_evalue"] / users_df["nb_questions"]
    users_df.reset_index(inplace=True)

    return users_df


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

def build_users_satisfaction_over_nb_eval(users_df:pd.DataFrame) ->pd.DataFrame:
    users_df["evalue"] = users_df["nb_questions"] - users_df["non_evalue"]
    users_df.sort_values(by="evalue", ascending=True, inplace=True)
    users_df["eval_mean"] = (users_df["great"] + 2. * users_df["ok"] / 3. + users_df["missing_info"] / 3.) / users_df["evalue"]
    users_df["eval_mean_std"] =  users_df["eval_mean"]
    users_df["eval_std"] = np.sqrt((users_df["great"]*(1 - users_df["eval_mean"])**2 + \
        users_df["ok"] * (2./3. - users_df["eval_mean"])**2 + \
        users_df["missing_info"] * (1. / 3. - users_df["eval_mean"])**2)/users_df["evalue"])
    users_satisfaction_over_nb_eval = pd.pivot_table(
        data=users_df,
        index="evalue",
        values=["eval_mean", "eval_mean_std", "eval_std"],
        aggfunc={
            "eval_mean": "mean",
            "eval_mean_std": "std", 
            "eval_std": "mean"
        }        
    )

    return users_satisfaction_over_nb_eval


def display_users_satisfaction_over_nb_eval(users_satisfaction:pd.DataFrame) ->pd.DataFrame:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=[1./4, 1./4,],
            x = [users_satisfaction.index[0], users_satisfaction.index[-1]],
            mode="none",
            name="mauvais",
            # "red"
            fillcolor="rgba(255, 0, 0, 0.2)",
            stackgroup='one', # define stack group
            )       
        )
    fig.add_trace(
        go.Scatter(
            y=[1./4, 1./4,],
            x = [users_satisfaction.index[0], users_satisfaction.index[-1]],
            mode="none",
            name="info manquante",
            # "orange"
            fillcolor="rgba(255, 128, 0, 0.2)",
            stackgroup='one' # define stack group
            )       
        )
    fig.add_trace(
        go.Scatter(
            y=[1./4, 1./4,],
            x = [users_satisfaction.index[0], users_satisfaction.index[-1]],
            mode="none",
            name="ok",
            # "green"
            fillcolor="rgba(0, 255, 0, 0.2)",
            stackgroup='one' # define stack group
            )       
        )
    fig.add_trace(
        go.Scatter(
            y=[1./4, 1./4,],
            x = [users_satisfaction.index[0], users_satisfaction.index[-1]],
            mode="none",
            name="excellent",
            # "blue"
            fillcolor="rgba(0, 0, 255, 0.2)",
            stackgroup='one' # define stack group
            )       
        )
    fig.add_trace(
        go.Bar(
            y=users_satisfaction["eval_mean"],
            x = users_satisfaction.index,
            name="eval moyenne",
            marker_color="blue",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=users_satisfaction["eval_std"],
            x = users_satisfaction.index,
            mode="lines",
            name="variabilité moyenne par utilisateurs",
            line=dict(color="lightblue"),
            )       
        )
    fig.add_trace(
        go.Scatter(
            y=users_satisfaction["eval_mean_std"],
            x = users_satisfaction.index,
            mode="lines",
            name="variabilité moyenne entre utilisateurs",
            line=dict(color="lightgreen"),
            )
        )
    fig.update_layout(
        title="Evaluation moyenne des réponses par utilisateurs, en fonction du nb d'évaluation réalisée",
        xaxis_title="nb d'évaluations réalisées",
        yaxis_title="Evaluation moyenne, entre 0 et 1",
    )
    st.plotly_chart(fig)
    return


def display_user_hist_over_eval(users_df):

    distinct_count_with_na = users_df["evalue"].value_counts()


    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=distinct_count_with_na.index, 
        y = distinct_count_with_na.values
        )
    )
    fig.update_layout(title='Histogramme des utilisateurs par nb réponses évaluées', yaxis_title="nombre d'utilisateur", xaxis_title='nombre de questions évaluées')
    st.plotly_chart(fig)
    return


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
            display_indicators(filtered_df=filtered_df)

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
            users_satisfaction = build_users_satisfaction_over_nb_eval(users_df=users_df)
            display_users_satisfaction_over_nb_eval(users_satisfaction=users_satisfaction)
            display_user_hist_over_eval(users_df=users_df)

        with st.expander("Users raw data", expanded=False):
            st.write(users_df)


        # TODO : Analyse par conversations, analyse par chunks


main()
