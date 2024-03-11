import os
import time
import json
from datetime import datetime
import pandas as pd
from typing import List

import streamlit as st
from loguru import logger
from watchpoints import watch

from wattelse.chatbot.backend.backend import ChatbotBackEnd
from wattelse.chatbot.chat_history import ChatHistory
from wattelse.chatbot import (
    RETRIEVAL_DENSE,
    RETRIEVAL_BM25,
    RETRIEVAL_HYBRID,
    RETRIEVAL_HYBRID_RERANKER,
    retriever_config,
    generator_config,
    DATA_DIR,
    USER_MODE,
    USER_NAME,
)
from wattelse.chatbot.backend import FASTCHAT_LLM, OLLAMA_LLM, CHATGPT_LLM
from wattelse.chatbot.indexer import index_files
from wattelse.common import TEXT_COLUMN, FILENAME_COLUMN
from wattelse.chatbot.utils import highlight_answer
from wattelse.api.prompts import FR_USER_BASE_RAG, FR_USER_MULTITURN_RAG


@st.cache_resource
def initialize_backend(**kwargs):
    """Initializes a backend based on the 'right' parameters"""
    return ChatbotBackEnd(**kwargs)


def on_options_change(frame, elem, exec_info):
    """Callback function used by watch()"""
    st.session_state["backend"] = initialize_backend(
        **retriever_config, **generator_config
    )
    if st.session_state.get("data_files_from_parsing"):
        st.session_state["backend"].initialize_data(
            st.session_state["data_files_from_parsing"]
        )
    elif st.session_state.get("selected_files"):
        st.session_state["backend"].initialize_data(
            [DATA_DIR / sf for sf in st.session_state["selected_files"]]
        )


watch(retriever_config, generator_config, callback=on_options_change)

# Initialize st.session_state
if "prev_selected_file" not in st.session_state:
    st.session_state["prev_selected_file"] = None
if "data_files_from_parsing" not in st.session_state:
    st.session_state["data_files_from_parsing"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ChatHistory()
if "backend" not in st.session_state:
    st.session_state["backend"] = initialize_backend(
        **retriever_config, **generator_config
    )


def initialize_options_from_config():
    keypairs = retriever_config | generator_config
    for k, v in keypairs.items():
        # set session values
        if k not in st.session_state:
            st.session_state[k] = v


def update_config_from_gui():
    for k in [
        "reranker_model_name",
        "top_n_extracts",
        "retrieval_mode",
        "similarity_threshold",
        "use_cache",
    ]:
        retriever_config[k] = st.session_state.get(k)
    for k in [
        "llm_api_name",
        "expected_answer_size",
        "provide_explanations",
        "custom_prompt",
        "remember_recent_messages",
    ]:
        generator_config[k] = st.session_state.get(k)


def add_user_message_to_session(prompt):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def check_data():
    """Check if a file is uploaded"""
    if not st.session_state.get("selected_files") and not st.session_state.get(
        "uploaded_files"
    ):
        st.error("Select data files", icon="🚨")
        st.stop()


def generate_answer(query: str):
    """Generate an answer based on a query using the backend oracle"""
    # Check if a file is uploaded
    check_data()

    # Query the backend
    (
        st.session_state["relevant_extracts"],
        st.session_state["relevant_extracts_similarity"],
        streaming_answer,
    ) = st.session_state["backend"].query_oracle(
        query,
        st.session_state["chat_history"].get_recent_history(),
        **retriever_config,
        **generator_config,
    )
    return streaming_answer

def display_streaming_answer(streaming_answer, message_placeholder):
    """Use the streaming_answer object to print the streaming answer in the message_placeholder"""
    answer = ""
    for chunk in streaming_answer: # each API has a distinct behaviour
        if st.session_state["llm_api_name"] == FASTCHAT_LLM:
            answer += chunk.choices[0].text
        elif st.session_state["llm_api_name"] == OLLAMA_LLM:
            # Last streamed chunk is always incomplete. Catch it and remove it.
            # TODO : why is last chunk not complete only in Streamlit ?
            # The streaming answer works well outside Streamlit...
            try:
                answer += json.loads(chunk.decode("utf-8"))["response"]
            except Exception as e:
                logger.error(e)
        elif st.session_state["llm_api_name"] == CHATGPT_LLM:
            answer = chunk.choices[0].delta.content
            if answer:
                answer += answer
        message_placeholder.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

def display_relevant_extracts(answer: str, relevant_extracts: List[str], relevant_extracts_similarity: List[float]):
    highlighted_relevant_extracts = highlight_answer(answer, relevant_extracts)
    for extract, sim in zip(
        highlighted_relevant_extracts, relevant_extracts_similarity
    ):
        with st.chat_message("explanation", avatar="🔑"):
            # Add score to text explanation
            score = round(sim * 5) * "⭐"
            expl = extract[TEXT_COLUMN]
            expl = expl.replace("\n", "\n\n")
            st.markdown(f"{score}\n{expl}", unsafe_allow_html=True)
            # Add filename if available
            filename = extract.get(FILENAME_COLUMN)
            if filename:
                st.markdown(f"*Source: [{filename}]()*")

def on_file_change():
    if st.session_state.get("uploaded_files"):
        index_files()  # this will update st.session_state["data_files_from_parsing"]
        if st.session_state.get("data_files_from_parsing"):
            logger.debug("Data file changed! Resetting chat history")
            st.session_state["backend"].initialize_data(
                st.session_state["data_files_from_parsing"]
            )
            st.session_state["prev_selected_file"] = st.session_state[
                "data_files_from_parsing"
            ]
            reset_messages_history()

    elif st.session_state["selected_files"] != st.session_state["prev_selected_file"]:
        logger.debug("Data file changed! Resetting chat history")
        st.session_state["backend"].initialize_data(
            [DATA_DIR / sf for sf in st.session_state["selected_files"]]
        )
        st.session_state["prev_selected_file"] = st.session_state["selected_files"]
        reset_messages_history()


def display_dev_side_bar():
    """Side bar used if USER_MODE=False"""
    with st.sidebar:
        with st.form("parameters_sidebar"):
            #            if not st.session_state.get("llm_api"):
            #                st.session_state["llm_api"] = initialize_llm_api_wrapper(st.session_state.get("llm_api_name", LOCAL_LLM))
            #            st.markdown(f"**API** : *{st.session_state['llm_api'].model_name}*")
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

            with st.expander("Retriever configuration"):
                te1, te2 = st.tabs(["Embedding model", "Reranker model"])
                with te1:
                    # Embedding model
                    st.write(f'EmbeddingAPI: {st.session_state["backend"].embedding_api.get_api_model_name()}')
                with te2:
                    # Reranker model
                    st.selectbox(
                        "Reranker model",
                        [
                            "antoinelouis/crossencoder-camembert-base-mmarcoFR",
                            "dangvantuan/CrossEncoder-camembert-large",
                        ],
                        key="reranker_model_name",
                    )

                # Number of extracts to be considered
                st.slider(
                    "Top n extracts",
                    min_value=1,
                    max_value=10,
                    step=1,
                    key="top_n_extracts",
                )

                # Balance between dense and bm25 retrieval
                st.selectbox(
                    "Retrieval mode",
                    [
                        RETRIEVAL_BM25,
                        RETRIEVAL_DENSE,
                        RETRIEVAL_HYBRID,
                        RETRIEVAL_HYBRID_RERANKER,
                    ],
                    key="retrieval_mode",
                )

                # Similarity threshold
                st.slider(
                    "Similarity threshold for extracts",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    key="similarity_threshold",
                )

                # Use cache
                st.toggle(
                    "Use cache",
                    key="use_cache",
                )

            with st.expander("LLM generator configuration"):
                # answer size
                st.selectbox(
                    "Answer size",
                    ["short", "detailed"],
                    key="expected_answer_size",
                )

                # Custom prompt
                st.text_area("Prompt", key="custom_prompt")

                # Choice of LLM API
                st.selectbox(
                    "LLM API type",
                    [FASTCHAT_LLM, OLLAMA_LLM, CHATGPT_LLM],
                    key="llm_api_name",
                    index=0,
                )

            parameters_sidebar_clicked = st.form_submit_button("Apply", type="primary")

            if parameters_sidebar_clicked:
                logger.debug("Parameters saved!")
                check_data()
                update_config_from_gui()

                on_file_change()
                # st.session_state["data_files_from_parsing"] = [] # remove all previous files

                info = st.info("Parameters saved!")
                time.sleep(0.5)
                info.empty()  # Clear the alert

        # Memory management
        st.toggle(
            "Use recent interaction history",
            value=False,
            key="remember_recent_messages",
            on_change=switch_prompt,
        )

        display_buttons()

def display_user_side_bar():
    """Side bar used if USER_MODE=True"""
    with st.sidebar:
        # Show user
        st.write(f'**User:** {USER_NAME}')
        # Show selected files
        st.header("Loaded files :")
        for file in st.session_state["selected_files"]:
            st.write("- " + file.split(".")[0])
        st.markdown("---")
        # Add button here as in Streamlit you can't add button below input_chat widget...
        st.button("Clear history", on_click=reset_messages_history)
        


def switch_prompt():
    """Switch prompt between history and non history versions"""
    if st.session_state["remember_recent_messages"]:
        st.session_state["custom_prompt"] = FR_USER_MULTITURN_RAG
    else:
        st.session_state["custom_prompt"] = FR_USER_BASE_RAG
    update_config_from_gui()


def display_buttons():
    col1, col2 = st.columns(2, gap="small")
    with col1:

        def reset_prompt():
            st.session_state["custom_prompt"] = FR_USER_BASE_RAG

        st.button("Reset prompt", on_click=reset_prompt)
    with col2:
        st.button("Clear history", on_click=reset_messages_history)


def reset_messages_history():
    # clear messages
    st.session_state["messages"] = []
    # clean database
    st.session_state["chat_history"].db_table.truncate()
    logger.debug("History now empty")

def feedback_button():
    """Feedback button for user mode"""
    # Align left
    _, _, col3, col4 = st.columns([3,3,1,1])
    with col3:
        st.button(":+1:", use_container_width=True, on_click=save_feedback, args=(1,))
    with col4:
        st.button(":-1:", use_container_width=True, on_click=save_feedback, args=(-1,))

def save_feedback(score: int):
    """Save feedback in the user directory"""
    collected_data = {
        "timestamp": datetime.today(),
        "query": st.session_state["messages"][-2]["content"],
        "answer": st.session_state["messages"][-1]["content"],
        "extracts": [extract["text"] for extract in st.session_state["relevant_extracts"]], 
        "score": score,
        "embedding_model": st.session_state["backend"].embedding_api.get_api_model_name(),
        "generation_model": st.session_state["backend"].llm_api.get_api_model_name(),
        "retrieval_mode": st.session_state["retrieval_mode"],
        "files": st.session_state["selected_files"],
    }
    feedback_file_path = DATA_DIR.parent / "feedback.csv"
    pd.DataFrame([collected_data]).to_csv(feedback_file_path, mode="a", header=not feedback_file_path.is_file(), index=False)
    # Notify the user
    st.toast("Feedback saved !")


def main():
    # Title and initialization
    st.title("WattElse® Chat")
    initialize_options_from_config()

    # User or dev sidebar
    if USER_MODE:
        st.session_state["selected_files"] = os.listdir(DATA_DIR)
        display_user_side_bar()
        on_file_change()
    else:
        display_dev_side_bar()
        

    # Split chat and explanations
    t1, t2 = st.tabs(["Chat", "Explanations"])
    with t1 :
        display_existing_messages()
    
    # Query input bar
    query = st.chat_input("Enter any question in relation with the provided documents")

    if query:
        with t1:
            add_user_message_to_session(query)
            with st.chat_message("assistant"):
                # HAL answer GUI initialization

                message_placeholder = st.empty()
                message_placeholder.markdown("...")
                streaming_answer = generate_answer(query)
                display_streaming_answer(streaming_answer, message_placeholder)
                if USER_MODE:
                    feedback_button()
        with t2:
            display_relevant_extracts(
                st.session_state["messages"][-1]["content"],
                st.session_state["relevant_extracts"],
                st.session_state["relevant_extracts_similarity"]
                )
        st.session_state["chat_history"].add_to_database(query, st.session_state["messages"][-1]["content"])


if __name__ == "__main__":
    main()
