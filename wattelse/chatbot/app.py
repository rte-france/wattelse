import os
import time
import json

import streamlit as st
from loguru import logger
from watchpoints import watch

from wattelse.chatbot.backend.backend import ChatbotBackEnd
from wattelse.chatbot.chat_history import ChatHistory
from wattelse.chatbot import RETRIEVAL_DENSE, RETRIEVAL_BM25, RETRIEVAL_HYBRID, \
    RETRIEVAL_HYBRID_RERANKER, FASTCHAT_LLM, OLLAMA_LLM, CHATGPT_LLM, retriever_config, generator_config, DATA_DIR
from wattelse.chatbot.indexer import index_files
from wattelse.common import TEXT_COLUMN, FILENAME_COLUMN
from wattelse.chatbot.utils import highlight_answer
from wattelse.llm.prompts import FR_USER_BASE_RAG, FR_USER_MULTITURN_RAG


@st.cache_resource
def initialize_backend(**kwargs):
    """Initializes a backend based on the 'right' parameters"""
    return ChatbotBackEnd(**kwargs)

def on_options_change(frame, elem, exec_info):
    st.session_state["backend"] = initialize_backend(**retriever_config, **generator_config)

watch(retriever_config, generator_config, callback=on_options_change)

# Initialize st.session_state
if "prev_selected_file" not in st.session_state:
    st.session_state["prev_selected_file"] = None
if "prev_embedding_model" not in st.session_state:
    st.session_state["prev_embedding_model"] = None
if "data_files_from_parsing" not in st.session_state:
    st.session_state["data_files_from_parsing"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ChatHistory()

if "backend" not in st.session_state:
    st.session_state["backend"] = initialize_backend(**retriever_config, **generator_config)


def initialize_options_from_config():
    keypairs = retriever_config | generator_config
    for k, v in keypairs.items():
        # set session values
        if k not in st.session_state:
            st.session_state[k] = v

def update_config_from_gui():
    for k in ["embedding_model_name", "reranker_model_name", "top_n_extracts",
              "retrieval_mode", "similarity_threshold", "use_cache"]:
        retriever_config[k] = st.session_state[k]
    for k in ["llm_api_name", "expected_answer_size", "provide_explanations",
              "custom_prompt", "remember_recent_messages"]:
        generator_config[k] = st.session_state[k]


def add_user_message_to_session(prompt):
    if prompt:
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
    if (st.session_state.get("selected_files") is None
            and st.session_state.get("data_files_from_parsing") is None):
        st.error("Select data files", icon="🚨")
        st.stop()


def generate_assistant_response(query):
    check_data()

    with st.chat_message("assistant"):
        # HAL answer GUI initialization
        message_placeholder = st.empty()
        message_placeholder.markdown("...")

        # Query the backend
        relevant_extracts, relevant_extracts_similarity, stream_response = st.session_state["backend"].query_oracle(query, st.session_state["chat_history"].get_history(), **retriever_config, **generator_config)

        # HAL final response
        response = ""
        for chunk in stream_response:
            if st.session_state["llm_api_name"]==FASTCHAT_LLM:
                response += chunk.choices[0].text
            elif st.session_state["llm_api_name"]==OLLAMA_LLM:
                # Last streamed chunk is always incomplete. Catch it and remove it.
                # TODO : why is last chunk not complete only in Streamlit ?
                # The streaming response works well outside Streamlit...
                try:
                    response += json.loads(chunk.decode('utf-8'))["response"]
                except Exception as e:
                    logger.error(e)
            elif st.session_state["llm_api_name"]==CHATGPT_LLM:
                answer = chunk.choices[0].delta.content
                if answer:
                    response += answer
            message_placeholder.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})

    highlighted_relevant_extracts = highlight_answer(response, relevant_extracts)
    if st.session_state["provide_explanations"]:
        with st.expander("Explanation"):
            for extract, sim in zip(highlighted_relevant_extracts, relevant_extracts_similarity):
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

    return response


def on_file_change():
    if st.session_state.get("uploaded_files"):
        index_files()  # this will update st.session_state["data_files_from_parsing"]
        if st.session_state.get("data_files_from_parsing"):
            logger.debug("Data file changed! Resetting chat history")
            st.session_state["backend"].initialize_data(st.session_state["data_files_from_parsing"])
            st.session_state["prev_selected_file"] = st.session_state[
                "data_files_from_parsing"
            ]
            reset_messages_history()

    elif (st.session_state["selected_files"] != st.session_state["prev_selected_file"]) or (st.session_state["prv_embedding_model_name"] != st.session_state["embedding_model_name"]):
        logger.debug("Data file changed! Resetting chat history")
        st.session_state["backend"].initialize_data([DATA_DIR / sf for sf in st.session_state["selected_files"]])
        st.session_state["prev_selected_file"] = st.session_state["selected_files"]
        st.session_state["prv_embedding_model_name"] = st.session_state["embedding_model_name"]
        reset_messages_history()


def on_instruct_prompt_change():
    logger.debug(
        f"New instruct prompt size: {st.session_state['expected_answer_size']}"
    )


def display_side_bar():
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
                key="use_cache",
            )

            # Relevant references as explanations
            st.toggle("Provide explanations", key="provide_explanations")

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
                [RETRIEVAL_BM25, RETRIEVAL_DENSE, RETRIEVAL_HYBRID, RETRIEVAL_HYBRID_RERANKER],
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

            # Custom prompt
            st.text_area("Prompt", key="custom_prompt")

            # Choice of LLM API
            st.selectbox(
                "LLM API type",
                [FASTCHAT_LLM, OLLAMA_LLM, CHATGPT_LLM],
                key="llm_api_name",
                index=0
            )

            parameters_sidebar_clicked = st.form_submit_button("Apply", type="primary")

            if parameters_sidebar_clicked:
                logger.debug("Parameters saved!")
                update_config_from_gui()

                st.session_state["data_files_from_parsing"] = [] # remove all previous files
                on_file_change()
                on_instruct_prompt_change()
                check_data()

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
            "Export discussion", data=st.session_state["chat_history"].export_history(), file_name="history.json"
        )


def reset_messages_history():
    # clear messages
    st.session_state["messages"] = []
    # clean database
    st.session_state["chat_history"].db_table.truncate()
    logger.debug("History now empty")


def main():
    st.title("WattElse® Chat")
    # st.markdown("**W**ord **A**nalysis and **T**ext **T**racking with an **E**nhanced **L**anguage model **S**earch **E**ngine")
    st.markdown(
        "**W**holistic **A**nalysis of  **T**ex**T** with an **E**nhanced **L**anguage model **S**earch **E**ngine"
    )

    initialize_options_from_config()

    display_side_bar()

    display_existing_messages()

    query = st.chat_input("Enter any question in relation with the provided document")
    if query:
        add_user_message_to_session(query)

        response = generate_assistant_response(query)
        st.session_state["chat_history"].add_to_database(query, response)

    display_reset()


if __name__ == "__main__":
    main()
