import datetime
import json
import os
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

from wattelse.common import TEXT_COLUMN, FILENAME_COLUMN
from wattelse.common.text_parsers.extract_text_from_MD import parse_md
from wattelse.common.text_parsers.extract_text_from_PDF import parse_pdf
from wattelse.common.text_parsers.extract_text_using_origami import parse_docx
from wattelse.chatbot.utils import (
    extract_n_most_relevant_extracts,
    generate_RAG_prompt,
    load_data,
    make_docs_BM25_indexing,
)
from wattelse.common.vars import BASE_DATA_DIR
from wattelse.llm.vars import TEMPERATURE
from wattelse.llm.vllm_api import vLLM_API
from wattelse.llm.prompts import FR_USER_BASE_RAG, FR_USER_MULTITURN_RAG, FR_USER_MULTITURN_QUESTION_SPECIFICATION
from sentence_transformers import SentenceTransformer, CrossEncoder

DATA_DIR = BASE_DATA_DIR / "chatbot"
# Make dirs if not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ensures to write with +rw for both user and groups
os.umask(0o002)

# inspired by: https://github.com/mobarski/ask-my-pdf &  https://github.com/cefege/seo-chat-bot/blob/master/streamlit_app.py

DEFAULT_MEMORY_DELAY = 2  # in minutes

# Generation model API parameters
API = vLLM_API()
MAX_TOKENS = 512

# Embedding model parameters
EMBEDDING_MODEL_NAME = "antoinelouis/biencoder-camembert-base-mmarcoFR"

if "prev_selected_file" not in st.session_state:
    st.session_state["prev_selected_file"] = None
if "prev_embedding_model" not in st.session_state:
    st.session_state["prev_embedding_model"] = None
if "data_files_from_parsing" not in st.session_state:
    st.session_state["data_files_from_parsing"] = []

@st.cache_resource
def initialize_embedding_model(embedding_model_name):
    """Load embedding_model"""
    logger.info("Initializing embedding model...")
    embedding_model = SentenceTransformer(embedding_model_name)
    # Fix model max input length issue
    if embedding_model.max_seq_length == 514:
        embedding_model.max_seq_length = 512
    return embedding_model

@st.cache_resource
def initialize_reranker_model(reranker_model_name):
    """Load embedding_model and reranker_model"""
    logger.info("Initializing embedding and reranker models...")
    reranker_model = CrossEncoder(reranker_model_name)
    return reranker_model


def initialize_data(data_path: Path, embedding_model, embedding_model_name, use_cache=True):
    if not data_path.is_file():
        st.error("Select a data file", icon="🚨")
    data, docs_embeddings = load_data(data_path,
                                      embedding_model,
                                      embedding_model_name=embedding_model_name,
                                      use_cache=use_cache,
                                      )
    st.session_state["data"] = data
    st.session_state["docs_embeddings"] = docs_embeddings
    return data, docs_embeddings

def initialize_data_list(data_paths: List[Path], embedding_model, embedding_model_name, use_cache=True):
    data_l = None
    embs_a = None
    for data_path in data_paths:
        data, embs = initialize_data(data_path, embedding_model, embedding_model_name, use_cache)
        data_l = data if data_l is None else pd.concat([data_l, data], axis=0).reset_index(drop=True)
        embs_a = embs if embs_a is None else np.concatenate((embs_a, embs))
    st.session_state["docs"] = docs_l
    st.session_state["docs_embeddings"] = embs_a

@st.cache_resource
def initialize_db():
    # in memory small DB for handling messages
    db = TinyDB(storage=MemoryStorage)
    table = db.table("messages")
    return table


# initialize DB
db_table = initialize_db()


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
    history = get_history()
    enriched_query = API.generate(FR_USER_MULTITURN_QUESTION_SPECIFICATION.format(history=history, query=query),
                                  temperature=TEMPERATURE,
                                  max_tokens=MAX_TOKENS,
                                  )
    return enriched_query

def get_history():
    """Return the history of the conversation"""
    context = get_recent_context()
    history = ""
    for entry in context:
        history += "Utilisateur : {query}\nRéponse : {response}\n".format(query=entry["query"], response=entry["response"])
    return history

def generate_assistant_response(query, embedding_model):
    if st.session_state.get("data") is None:
        st.error("Select a document first", icon="🚨")
        return
        
    history = ""
    enriched_query = query
    if st.session_state["remember_recent_messages"]:
        enriched_query = enrich_query(query)
        logger.debug(enriched_query)
        history = get_history()
        
    relevant_extracts, relevant_extracts_similarity = extract_n_most_relevant_extracts(
        st.session_state["top_n_extracts"],
        enriched_query,
        st.session_state["data"],
        st.session_state["docs_embeddings"],
        embedding_model,
        st.session_state["bm25_model"],
        retrieval_mode=st.session_state["retrieval_mode"],
        reranker_model=st.session_state.get("reranker_model"),
        similarity_threshold=st.session_state["similarity_threshold"],
    )

    with st.chat_message("assistant"):
        # HAL answer GUI initialization
        message_placeholder = st.empty()
        message_placeholder.markdown("...")

        # Generates prompt
        prompt = generate_RAG_prompt(
            query,
            [extract[TEXT_COLUMN] for extract in relevant_extracts],
            expected_answer_size=st.session_state["expected_answer_size"],
            custom_prompt=st.session_state["custom_prompt"],
            history=history,
        )
        logger.debug(f"Prompt : {prompt}")
        # Generates response
        stream_response = API.generate(prompt,
                                       #system_prompt=FR_SYSTEM_DODER_RAG, -> NOT WORKING WITH CERTAIN MODELS (MISTRAL)
                                       temperature=TEMPERATURE,
                                       max_tokens=MAX_TOKENS,
                                       stream=True)

        # HAL final response
        response = ""
        for chunk in stream_response:
            response += chunk.choices[0].text
            message_placeholder.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})

    if st.session_state["provide_explanations"]:
        with st.expander("Explanation"):
            for extract, sim in zip(relevant_extracts, relevant_extracts_similarity):
                with st.chat_message("explanation", avatar="🔑"):
                    # Add score to text explanation
                    score = round(sim * 5) * "⭐"
                    expl = extract[TEXT_COLUMN]
                    expl = expl.replace("\n", "\n\n")
                    st.write(f"{score}\n{expl}")
                    # Add filename if available
                    filename = extract.get(FILENAME_COLUMN)
                    if filename:
                        st.markdown(f"*Source: [{filename}]()*")

    return response


def add_to_database(query, response):
    timestamp = datetime.datetime.now()
    db_table.insert({"query": query, "response": response, "timestamp": timestamp})


def get_recent_context(delay=DEFAULT_MEMORY_DELAY):
    """Returns a list of recent answers from the bot that occured during the indicated delay in minutes"""
    current_timestamp = datetime.datetime.now()
    q = Query()
    return db_table.search(
        q.timestamp > (current_timestamp - datetime.timedelta(minutes=delay))
    )

def index_file(uploaded_file: UploadedFile):
    # NB: UploadedFile  can be used like a BytesIO
    extension = uploaded_file.name.split(".")[-1].lower()
    # create a temporary file using a context manager
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(tmpdirname + "/" + uploaded_file.name, "wb") as file:
            # store temporary file
            file.write(uploaded_file.getbuffer())
            # extract data from file
            if extension == "pdf":
                st.session_state["data_files_from_parsing"].append(parse_pdf(
                    Path(file.name), DATA_DIR)
                )
            elif extension == "docx":
                st.session_state["data_files_from_parsing"].append(parse_docx(
                    Path(file.name), DATA_DIR
                ))
            elif extension == "md":
                st.session_state["data_files_from_parsing"].append(parse_md(
                    Path(file.name), DATA_DIR
                ))
            else:
                st.error("File type not supported!")

def index_files():
    uploaded_files = st.session_state["uploaded_files"]
    logger.debug(f"Uploading file: {uploaded_files}")
    for f in uploaded_files:
        index_file(f)


def on_file_change():
    if st.session_state.get("uploaded_files"):
        index_files()  # this will update st.session_state["data_files_from_parsing"]
        if st.session_state.get("data_files_from_parsing"):
            logger.debug("Data file changed! Resetting chat history")
            initialize_data_list(st.session_state["data_files_from_parsing"],
                            st.session_state["embedding_model"],
                            embedding_model_name=st.session_state["embedding_model_name"],
                            use_cache=st.session_state["use_cache"],
                            )
            st.session_state["prev_selected_file"] = st.session_state[
                "data_files_from_parsing"
            ]
            reset_messages_history()

    elif (st.session_state["selected_files"] != st.session_state["prev_selected_file"]) or (st.session_state["prv_embedding_model"] != st.session_state["embedding_model"]):
        logger.debug("Data file changed! Resetting chat history")
        initialize_data_list([DATA_DIR / sf for sf in st.session_state["selected_files"]],
                             st.session_state["embedding_model"],
                             embedding_model_name=st.session_state["embedding_model_name"],
                             use_cache=st.session_state["use_cache"],
                        )
        st.session_state["prev_selected_file"] = st.session_state["selected_files"]
        st.session_state["prv_embedding_model"] = st.session_state["embedding_model"]
        reset_messages_history()


def on_instruct_prompt_change():
    logger.debug(
        f"New instruct prompt size: {st.session_state['expected_answer_size']}"
    )


def display_side_bar():
    with st.sidebar:
        with st.form("parameters_sidebar"):
            st.markdown(f"**API** : *{API.model_name}*")
            st.title("Parameters")

            # Data
            t1, t2 = st.tabs(["SELECT", "UPLOAD"])
            with t1:
                data_options = [""] + sorted(os.listdir(DATA_DIR))
                st.multiselect(
                    "Select input data",
                    data_options,
                    key="selected_files",
                    disabled=bool(st.session_state.get("uploaded_files")),
                )
            with t2:
                st.file_uploader(
                    "File",
                    type=["pdf", "docx", "md"],
                    key="uploaded_files",
                    label_visibility="collapsed",
                    accept_multiple_files=True,
                )
            
            # Embedding model
            st.selectbox(
                "Embedding model",
                ["antoinelouis/biencoder-camembert-base-mmarcoFR",
                "dangvantuan/sentence-camembert-large"],
                key="embedding_model_name",
            )

            # Reranker model
            st.selectbox(
                "Reranker model",
                ["antoinelouis/crossencoder-camembert-base-mmarcoFR",
                 "dangvantuan/CrossEncoder-camembert-large"],
                key="reranker_model_name",
            )

            # Response size
            st.selectbox(
                "Response size",
                ["short", "detailed"],
                key="expected_answer_size",
            )

            # Use cache
            st.toggle(
                "Use cache",
                value=True,
                key="use_cache",
            )

            # Relevant references as explanations
            st.toggle("Provide explanations", value=True, key="provide_explanations")

            # Number of extracts to be considered
            st.slider(
                "Top n extracts",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key="top_n_extracts",
            )

            # Balance between dense and bm25 retrieval
            st.selectbox(
                "Retrieval mode",
                ["bm25", "dense", "hybrid", "hybrid+reranker"],
                index=2,
                key="retrieval_mode",
            )

            # Similarity threshold
            st.slider(
                "Similarity threshold for extracts",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="similarity_threshold",
            )

            # Custom prompt
            st.text_area("Prompt", value=FR_USER_BASE_RAG, key="custom_prompt")

            parameters_sidebar_clicked = st.form_submit_button("Apply")

            if parameters_sidebar_clicked:
                logger.debug("Parameters updated!")
                st.session_state["embedding_model"] = initialize_embedding_model(st.session_state["embedding_model_name"])
                if st.session_state["retrieval_mode"] == "hybrid+reranking":
                    st.session_state["reranker_model"] = initialize_reranker_model(st.session_state["reranker_model_name"])
                st.session_state["data_files_from_parsing"] = [] # remove all previous files
                on_file_change()
                on_instruct_prompt_change()
                if st.session_state["retrieval_mode"] in ("bm25", "hybrid", "hybrid+reranker"):
                    st.session_state["bm25_model"] = make_docs_BM25_indexing(st.session_state["docs"])
                info = st.info("Parameters saved!")
                time.sleep(0.5)
                info.empty()  # Clear the alert
        # Memory management
        st.toggle("Use recent interaction history",
                value=False,
                key="remember_recent_messages",
                on_change=switch_prompt,
                )

def switch_prompt():
    """Switch prompt between hisotry and non history versions"""
    if st.session_state["remember_recent_messages"]:
        st.session_state["custom_prompt"] = FR_USER_MULTITURN_RAG
    else:
        st.session_state["custom_prompt"] = FR_USER_BASE_RAG

def display_reset():
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:

        def reset_prompt():
            st.session_state["custom_prompt"] = FR_USER_BASE_RAG

        st.button("Reset prompt", on_click=reset_prompt)
    with col2:
        st.button("Clear discussion", on_click=reset_messages_history)
    with col3:
        st.download_button(
            "Export discussion", data=export_history(), file_name="history.json"
        )


def main():
    st.title("WattElse® Chat")
    # st.markdown("**W**ord **A**nalysis and **T**ext **T**racking with an **E**nhanced **L**anguage model **S**earch **E**ngine")
    st.markdown(
        "**W**holistic **A**nalysis of  **T**ex**T** with an **E**nhanced **L**anguage model **S**earch **E**ngine"
    )
    display_side_bar()

    display_existing_messages()

    query = st.chat_input("Enter any question in relation with the provided document")
    if query:
        add_user_message_to_session(query)

        response = generate_assistant_response(query, st.session_state["embedding_model"])
        add_to_database(query, response)

    display_reset()


if __name__ == "__main__":
    main()
