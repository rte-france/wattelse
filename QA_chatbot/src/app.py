import datetime
import os
from pathlib import Path

import streamlit as st
from loguru import logger
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

from chatbot import N, initialize_models, load_data
from utils import extract_n_most_relevant_extracts, generate_answer

DATA_DIR = "./data"
DEFAULT_MEMORY_DELAY = 2 # in minutes

# inspiration: https://github.com/mobarski/ask-my-pdf &  https://github.com/cefege/seo-chat-bot/blob/master/streamlit_app.py

if "prev_selected_file" not in st.session_state:
    st.session_state["prev_selected_file"] = None

@st.cache_resource
def initialize():
    """Streamlit wrapper to manage data caching"""
    return initialize_models()


@st.cache_resource
def initialize_data(data_name: str):
    data_path = Path(DATA_DIR) / data_name  # TODO choose
    docs, docs_embeddings = load_data(data_path, embedding_model, use_cache=True)
    st.session_state["docs"] = docs
    st.session_state["docs_embeddings"] = docs_embeddings
    return docs, docs_embeddings

@st.cache_resource
def initialize_db():
    # in memory small DB for handling messages
    db = TinyDB(storage=MemoryStorage)
    table = db.table("messages")
    return table

# initialize models
embedding_model, tokenizer, instruct_model = initialize()

# initialize DB
db_table = initialize_db()

def export_history():
    """Export messages in JSON from the database"""
    logger.warning("Not implemented yet")


def add_user_message_to_session(prompt):
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)


def reset_messages_history():
    # clear messages
    st.session_state["messages"] = []
    # clean database
    db_table.truncate()


def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def enrich_query(query):
    """Use recent interaction context to enrich the user query"""
    context = get_recent_context()
    tips = ""
    for entry in context:
        tips += entry["response"] + " "
    enriched_query = tips + query
    logger.debug(f"Query with short-time context: {enriched_query}")
    return enriched_query

def generate_assistant_response(query):
    if st.session_state["remember_recent_messages"]:
        query = enrich_query(query)

    relevant_extracts = extract_n_most_relevant_extracts(
        N, query, st.session_state["docs"], st.session_state["docs_embeddings"], embedding_model
    )

    with st.chat_message("assistant"):
        # HAL answer GUI initialization
        message_placeholder = st.empty()
        message_placeholder.markdown("...")

        # Generation of response
        response = generate_answer(instruct_model, tokenizer, query, relevant_extracts, st.session_state["expected_answer_size"])

        # HAL final response
        message_placeholder.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})

    if st.session_state["provide_explanations"]:
        with st.chat_message("explanation", avatar="🔑"):
            explanation_placeholder = st.empty()
            explanations = "Blablabla"
            explanation_placeholder.markdown(f"```{explanations}``")

    return response


def add_to_database(query, response):
    timestamp = datetime.datetime.now()
    db_table.insert({"query": query, "response": response, "timestamp": timestamp})
    logger.debug("Database updated")


def get_recent_context(delay = DEFAULT_MEMORY_DELAY):
    """Returns a list of recent answers from the bot that occured during the indicated delay in minutes"""
    current_timestamp = datetime.datetime.now()
    q = Query()
    return db_table.search(q.timestamp > (current_timestamp - datetime.timedelta(minutes = delay)))

def index_file():
    logger.debug(f"Uploading file: {st.session_state['uploaded_file']}")
    logger.warning("Not implemented yet!")

def on_file_change():
    if st.session_state["selected_file"] != st.session_state["prev_selected_file"]:
        logger.debug("Data file changed! Resetting chat history")
        initialize_data(st.session_state["selected_file"])
        st.session_state["prev_selected_file"] = st.session_state["selected_file"]
        reset_messages_history()


def on_instruct_prompt_change():
    logger.debug(
        f"New instruct prompt size: {st.session_state['expected_answer_size']}"
    )

def display_side_bar():
    with st.sidebar.form("parameters_sidebar"):
        st.title("Parameters")

        # Data
        t1, t2 = st.tabs(["SELECT", "UPLOAD"])
        with t1:
            data_options = [""] + os.listdir(DATA_DIR)
            st.selectbox(
                "Select input data",
                data_options,
                #on_change=on_file_change,
                key="selected_file",
            )
        with t2:
            st.file_uploader(
                "File",
                type=["pdf", "docx", "txt", "md"],
                key="uploaded_file",
                #on_change=index_file,
                label_visibility="collapsed",
            )

        # Response size
        st.selectbox(
            "Response size",
            ["short", "detailed"],
            #on_change=on_instruct_prompt_change,
            key="expected_answer_size",
        )

        # Memory management
        st.checkbox("Use recent interaction history", value=True, key="remember_recent_messages")

        # Relevant references as explanations
        st.checkbox("Provide explanations", value=False, key="provide_explanations")
        
        parameters_sidebar_clicked = st.form_submit_button("Apply")
        if parameters_sidebar_clicked:
            logger.debug("Parameters changed!")
            on_file_change()
            on_instruct_prompt_change()

def display_reset():
    st.button("Clear discussion", on_click=reset_messages_history)
    st.button("Export discussion", on_click=export_history)


def main():
    st.title("WattChat®")

    display_side_bar()

    logger.debug("Ready to chat!")

    display_existing_messages()

    query = st.chat_input("Enter any question in relation with the provided document")
    if query:
        add_user_message_to_session(query)

        response = generate_assistant_response(query)

        add_to_database(query, response)

    display_reset()


if __name__ == "__main__":
    main()
