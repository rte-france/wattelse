#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import pandas as pd
import streamlit as st
import sqlite3


from pathlib import Path

from wattelse.chatbot.frontend.django_chatbot.settings import DB_DIR

DB_PATH = DB_DIR / "db.sqlite3"
DATA_TABLE = "chatbot_chat"
USER_TABLE = "auth_user"


def get_db_data(path_to_db: Path) -> pd.DataFrame:
    con = sqlite3.connect(path_to_db)

    # Get column names from the table
    cur = con.cursor()
    cur.execute(f"SELECT username, group_id, conversation_id, message, response, timestamp, "
                f"short_feedback, long_feedback "
                f"FROM {DATA_TABLE}, {USER_TABLE} "
                f"WHERE {DATA_TABLE}.user_id = {USER_TABLE}.id")
    column_names = [desc[0] for desc in cur.description]  # Get column names from description
    data = cur.fetchall()

    # Create DataFrame with column names
    df = pd.DataFrame(data, columns=column_names)

    con.close()
    return df


def side_bar():
    ### SIDEBAR OPTIONS ###
    with st.sidebar.form("parameters_sidebar"):
        st.title("Parameters")
        parameters_sidebar_clicked = st.form_submit_button(
        "Save", type="primary",
        )

def main():
    # Wide layout
    st.set_page_config(page_title="Wattelse dashboard", layout="wide")

    ### TITLE ###
    st.title("Wattelse dashboard")

    side_bar()

main()