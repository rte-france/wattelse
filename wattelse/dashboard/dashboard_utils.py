#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import hmac
from datetime import date
from pathlib import Path
import sqlite3
import yaml

import streamlit as st
import pandas as pd

from wattelse.web_app.web_app.settings import DB_DIR

DB_PATH = DB_DIR / "db.sqlite3"

DATA_TABLE_RAG = "chatbot_chat"
DATA_TABLE_GPT = "chatbot_gptchat"
USER_TABLE = "auth_user"

DATA_TABLES = {"RAG": DATA_TABLE_RAG, "SecureGPT": DATA_TABLE_GPT}

# Get main experimentations group names
DRH_GROUP_NAME = "DRH"
METIERS_GROUP_NAME = "Expé_Métiers"

# Get Expé_Métiers group names list
TEST_GROUP_NAMES_LIST_FILE_PATH = Path(__file__).parent / "test_groups_list.yaml"
with open(TEST_GROUP_NAMES_LIST_FILE_PATH) as f:
    TEST_GROUP_NAMES_LIST = yaml.safe_load(f)


def get_db_data(
    path_to_db: Path = DB_PATH, data_tables: dict[str, str] = DATA_TABLES
) -> dict[str, pd.DataFrame]:
    """extract the django db of questions, answers and relevant_extracts. All is put in a dataframe

    Args:
        path_to_db (Path, optional): the path where the db can be found. Defaults to DB_PATH.
        data_tables (dict[str], optional): A dict containing the table_names in the streamlit and in the django_db. Defaults to DATA_TABLES.

    Returns:
        pd.DataFrame: the resulting dataframe
    """

    full_data = dict()
    con = sqlite3.connect(path_to_db)

    # Get column names from the table
    cur = con.cursor()
    for table_name, db_table in data_tables.items():

        query = f"SELECT username, group_id, conversation_id, message, response, answer_timestamp, answer_delay, short_feedback, long_feedback"

        if table_name == "RAG":
            query += ", relevant_extracts, group_system_prompt"
        query += (
            f" FROM {db_table}, {USER_TABLE} WHERE {db_table}.user_id = {USER_TABLE}.id"
        )

        cur.execute(query)
        column_names = [
            desc[0] for desc in cur.description
        ]  # Get column names from description
        data = cur.fetchall()

        # Create DataFrame with column names
        df = pd.DataFrame(data, columns=column_names)
        df.answer_timestamp = pd.to_datetime(df.answer_timestamp)
        df["answer_date"] = df.answer_timestamp.dt.date
        df["long_feedback_bool"] = (df["long_feedback"] != "").astype(int)

        full_data[table_name] = df

    con.close()

    return full_data


def initialize_state_session():
    """Initialise the streamlit state_session with default values."""
    if "selected_table" not in st.session_state:
        st.session_state["selected_table"] = list(DATA_TABLES)[0]
    if "full_data" not in st.session_state:
        st.session_state["full_data"] = get_db_data(
            path_to_db=DB_PATH, data_tables=DATA_TABLES
        )
        st.session_state["unfiltered_data"] = st.session_state["full_data"][
            st.session_state["selected_table"]
        ]
        st.session_state["filtered_data"] = st.session_state["unfiltered_data"]
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "group" not in st.session_state:
        st.session_state["group"] = None
    if "nb_reponse_lissage" not in st.session_state:
        st.session_state["nb_reponse_lissage"] = 15

    if "unfiltered_timestamp_range" not in st.session_state:
        # Select time range
        min_date = date.today()
        max_date = date(year=1900, month=1, day=1)
        for table in st.session_state["full_data"]:
            tmp_min_max = st.session_state["full_data"][table]["answer_date"].agg(
                ["min", "max"]
            )
            min_date = min(min_date, tmp_min_max["min"])
            max_date = max(max_date, tmp_min_max["max"])

        st.session_state["unfiltered_timestamp_range"] = (
            min_date,
            max_date,
        )
        st.session_state["unfiltered_timestamp_range"] = (
            min_date,
            max_date,
        )
    if "extract_substring" not in st.session_state:
        st.session_state["extract_substring"] = ""

    return


def update_state_session():
    """get the history dataframe from st.session_state["full_data"],
    filter it with the values selected in the side bar (left part of the screen),
    write the result in st.session_state["filtered_data"]
    """
    st.session_state["full_data"] = get_db_data(
        path_to_db=DB_PATH, data_tables=DATA_TABLES
    )
    # Select time range
    min_date = date.today()
    max_date = date(year=1900, month=1, day=1)
    for table in st.session_state["full_data"]:
        tmp_min_max = st.session_state["full_data"][table]["answer_date"].agg(
            ["min", "max"]
        )
        min_date = min(min_date, tmp_min_max["min"])
        max_date = max(max_date, tmp_min_max["max"])

    st.session_state["unfiltered_timestamp_range"] = (
        min_date,
        max_date,
    )

    st.session_state["unfiltered_data"] = st.session_state["full_data"][
        st.session_state["selected_table"]
    ]
    filtered = st.session_state["unfiltered_data"]
    if st.session_state["user"]:
        filtered = filtered[filtered.username == st.session_state["user"]]
    if st.session_state["group"]:
        if st.session_state["group"] == METIERS_GROUP_NAME:
            filtered = filtered[~filtered.group_id.isin(TEST_GROUP_NAMES_LIST)]
        else:
            filtered = filtered[filtered.group_id == st.session_state["group"]]

    # Filter dataset to select only text within time range
    timestamp_range = st.session_state["timestamp_range"]
    filtered = filtered.loc[
        (filtered["answer_date"] >= timestamp_range[0])
        & (filtered["answer_date"] <= timestamp_range[1])
    ]

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
