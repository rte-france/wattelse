import datetime
import json

import streamlit as st
from loguru import logger
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

DEFAULT_MEMORY_DELAY = 2  # in minutes


@st.cache_resource
def initialize_db():
    # in memory small DB for handling messages
    db = TinyDB(storage=MemoryStorage)
    table = db.table("messages")
    return table


db_table = initialize_db()


def get_history():
    """Return the history of the conversation"""
    context = get_recent_context()
    history = ""
    for entry in context:
        history += "Utilisateur : {query}\nRéponse : {response}\n".format(query=entry["query"], response=entry["response"])
    return history


def get_recent_context(delay=DEFAULT_MEMORY_DELAY):
    """Returns a list of recent answers from the bot that occured during the indicated delay in minutes"""
    current_timestamp = datetime.datetime.now()
    q = Query()
    return db_table.search(
        q.timestamp > (current_timestamp - datetime.timedelta(minutes=delay))
    )


def add_to_database(query, response):
    timestamp = datetime.datetime.now()
    db_table.insert({"query": query, "response": response, "timestamp": timestamp})


def export_history():
    """Export messages in JSON from the database"""
    return json.dumps(
        [
            {
                k: v if k != "timestamp" else v.strftime("%d-%m-%Y %H:%M:%S")
                for k, v in d.items()
            }
            for d in db_table.all()
        ]
    )


def reset_messages_history():
    # clear messages
    st.session_state["messages"] = []
    # clean database
    db_table.truncate()
    logger.debug("History now empty")
