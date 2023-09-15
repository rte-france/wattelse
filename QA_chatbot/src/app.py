import datetime
import json
import os
import socket
import tempfile
import time
from pathlib import Path

import streamlit as st
from loguru import logger
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

from chatbot import initialize_models, load_data
from extract_text_from_MD import parse_md
from extract_text_from_PDF import parse_pdf
from extract_text_using_origami import parse_docx
from utils import (
    USE_REMOTE_LLM_MODEL,
    extract_n_most_relevant_extracts,
    generate_answer_locally,
    generate_answer_remotely,
    generate_prompt,
)

DATA_DIR = (
    Path("/data/weak_signals/cache/chatbot_data")
    if socket.gethostname() == "groesplu0"
    else Path("./data")
)
# inspired by: https://github.com/mobarski/ask-my-pdf &  https://github.com/cefege/seo-chat-bot/blob/master/streamlit_app.py

DEFAULT_MEMORY_DELAY = 2 # in minutes


if "prev_selected_file" not in st.session_state:
    st.session_state["prev_selected_file"] = None

@st.cache_resource
def initialize():
    """Streamlit wrapper to manage data caching"""
    return initialize_models(USE_REMOTE_LLM_MODEL)


def initialize_data(data_path: Path):
    if not data_path.is_file():
        st.error("Select a data file", icon="🚨")
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
    return json.dumps(
        [{k:v if k!="timestamp" else v.strftime("%d-%m-%Y %H:%M:%S") for k, v in d.items()} for d in db_table.all()]
    )


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
    logger.debug("History now empty")


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
    if st.session_state.get("docs") is None:
        st.error("Select a document first", icon="🚨")
        return

    if st.session_state["remember_recent_messages"]:
        query = enrich_query(query)

    relevant_extracts, sims = extract_n_most_relevant_extracts(st.session_state["nb_extracts"],
                                                               query,
                                                               st.session_state["docs"],
                                                               st.session_state["docs_embeddings"],
                                                               embedding_model,
                                                               st.session_state["similarity_threshold"]
                                                               )

    if st.session_state["provide_explanations"]:
        response_col, explanations_col = st.columns([2,3], gap="small") # wide column for explanations
    else:
        response_col, explanations_col = st.columns([100,1], gap="small")

    with response_col:

        with st.chat_message("assistant"):
            # HAL answer GUI initialization
            message_placeholder = st.empty()
            message_placeholder.markdown("...")

            # Generates prompt
            prompt = generate_prompt(query, relevant_extracts, st.session_state["expected_answer_size"])

            # Generation of response
            if USE_REMOTE_LLM_MODEL:
                response = generate_answer_remotely(prompt)
            else:
                response = generate_answer_locally(
                    instruct_model, tokenizer, prompt
            )

            # HAL final response
            message_placeholder.markdown(response)

            st.session_state["messages"].append({"role": "assistant", "content": response})

    if st.session_state["provide_explanations"]:
        with explanations_col:
            for expl, sim in zip(relevant_extracts, sims):
                with st.chat_message("explanation", avatar="🔑"):
                    # Add score to text explanation
                    score = round(sim * 5) * "⭐"
                    st.caption(f"{score}\n{expl}")

    return response


def add_to_database(query, response):
    timestamp = datetime.datetime.now()
    db_table.insert({"query": query, "response": response, "timestamp": timestamp})


def get_recent_context(delay = DEFAULT_MEMORY_DELAY):
    """Returns a list of recent answers from the bot that occured during the indicated delay in minutes"""
    current_timestamp = datetime.datetime.now()
    q = Query()
    return db_table.search(q.timestamp > (current_timestamp - datetime.timedelta(minutes = delay)))


def index_file():
    uploaded_file = st.session_state["uploaded_file"]
    logger.debug(f"Uploading file: {uploaded_file}")

    if uploaded_file: # can be used like a BytesIO
        extension = uploaded_file.name.split(".")[-1].lower()
        # create a temporary file using a context manager
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(tmpdirname + "/" + uploaded_file.name, "wb") as file:
                # store temporary file
                file.write(uploaded_file.getbuffer())
                # extract data from file
                if extension == "pdf":
                    st.session_state["data_file_from_parsing"] = parse_pdf(Path(file.name), DATA_DIR)
                elif extension == "docx":
                    st.session_state["data_file_from_parsing"] = parse_docx(Path(file.name), DATA_DIR)
                elif extension == "md":
                    st.session_state["data_file_from_parsing"] = parse_md(Path(file.name), DATA_DIR)
                else:
                    st.error("File type not supported!")

def on_file_change():
    if st.session_state.get("uploaded_file") is not None:
        index_file() # this will update st.session_state["data_file_from_parsing"]
        if st.session_state.get("data_file_from_parsing") is not None:
            logger.debug("Data file changed! Resetting chat history")
            initialize_data(st.session_state["data_file_from_parsing"])
            st.session_state["prev_selected_file"] = st.session_state["data_file_from_parsing"]
            reset_messages_history()

    elif st.session_state["selected_file"] != st.session_state["prev_selected_file"]:
        logger.debug("Data file changed! Resetting chat history")
        initialize_data(DATA_DIR / st.session_state["selected_file"])
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
                key="selected_file",
                disabled = st.session_state.get("uploaded_file") is not None
            )
        with t2:
            st.file_uploader(
                "File",
                type=["pdf", "docx", "md"],
                key="uploaded_file",
                label_visibility="collapsed",
                accept_multiple_files = False
            )

        # Response size
        st.selectbox(
            "Response size",
            ["short", "detailed"],
            key="expected_answer_size",
        )

        # Memory management
        st.toggle("Use recent interaction history", value=True, key="remember_recent_messages")

        # Relevant references as explanations
        st.toggle("Provide explanations", value=False, key="provide_explanations")

        # Number of extracts to be considered
        st.slider("Maximum number of extracts used", min_value=1, max_value=10, value=5, step=1,
                  key="nb_extracts")

        # Similarity threshold
        st.slider("Similarity threshold for extracts", min_value=0., max_value=1., value=0.4, step=0.05,
                  key="similarity_threshold")

        parameters_sidebar_clicked = st.form_submit_button("Apply")

        if parameters_sidebar_clicked:
            logger.debug("Parameters updated!")
            on_file_change()
            on_instruct_prompt_change()
            info = st.info("Parameters saved!")
            time.sleep(0.5)
            info.empty()  # Clear the alert

def display_reset():
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.button("Clear discussion", on_click=reset_messages_history)
    with col2:
        st.download_button("Export discussion", data=export_history(), file_name="history.json")


def main():
    st.title("WattElse® Chat")
    #st.markdown("**W**ord **A**nalysis and **T**ext **T**racking with an **E**nhanced **L**anguage model **S**earch **E**ngine")
    st.markdown(
        "**W**holistic **A**nalysis of  **T**ex**T** with an **E**nhanced **L**anguage model **S**earch **E**ngine"
    )
    display_side_bar()

    display_existing_messages()

    query = st.chat_input("Enter any question in relation with the provided document")
    if query:
        add_user_message_to_session(query)

        response = generate_assistant_response(query)

        add_to_database(query, response)

    display_reset()


if __name__ == "__main__":
    main()
