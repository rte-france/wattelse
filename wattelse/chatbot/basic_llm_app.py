import time
import json

import streamlit as st
from loguru import logger
from watchpoints import watch

from wattelse.chatbot.app import (
    initialize_backend,
    initialize_options_from_config,
    display_buttons,
    update_config_from_gui,
    display_existing_messages,
    add_user_message_to_session,
    on_options_change,
)
from wattelse.chatbot.chat_history import ChatHistory
from wattelse.chatbot import (
    FASTCHAT_LLM,
    OLLAMA_LLM,
    CHATGPT_LLM,
    retriever_config,
    generator_config,
    MAX_TOKENS,
)
from wattelse.api.prompts import FR_USER_BASE_QUERY, FR_USER_BASE_MULTITURN_QUERY

watch(retriever_config, generator_config, callback=on_options_change)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ChatHistory()

if "backend" not in st.session_state:
    st.session_state["backend"] = initialize_backend(
        **retriever_config, **generator_config
    )


def generate_assistant_response(query):

    with st.chat_message("assistant"):
        # HAL answer GUI initialization
        message_placeholder = st.empty()
        message_placeholder.markdown("...")

        # Query the backend
        stream_response = st.session_state["backend"].simple_query(
            query,
            st.session_state["chat_history"].get_recent_history(),
            **generator_config,
        )

        # HAL final response
        response = ""
        for chunk in stream_response:
            if st.session_state["llm_api_name"] == FASTCHAT_LLM:
                response += chunk.choices[0].text
            elif st.session_state["llm_api_name"] == OLLAMA_LLM:
                # Last streamed chunk is always incomplete. Catch it and remove it.
                # TODO : why is last chunk not complete only in Streamlit ?
                # The streaming response works well outside Streamlit...
                try:
                    response += json.loads(chunk.decode("utf-8"))["response"]
                except Exception as e:
                    logger.error(e)
            elif st.session_state["llm_api_name"] == CHATGPT_LLM:
                answer = chunk.choices[0].delta.content
                if answer:
                    response += answer
            message_placeholder.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})

        return response


def switch_prompt():
    """Switch prompt between hisotry and non history versions"""
    if st.session_state["remember_recent_messages"]:
        st.session_state["custom_prompt"] = FR_USER_BASE_MULTITURN_QUERY
    else:
        st.session_state["custom_prompt"] = FR_USER_BASE_QUERY
    update_config_from_gui()


def display_side_bar():
    with st.sidebar:
        with st.form("parameters_sidebar"):
            st.title("Parameters")

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
                update_config_from_gui()

                info = st.info("Parameters saved!")
                time.sleep(0.5)
                info.empty()  # Clear the alert

    st.toggle(
        "Use recent interaction history",
        value=False,
        key="remember_recent_messages",
        on_change=switch_prompt,
    )


def main():
    st.title("WattElse® Basic LLM chat")
    # st.markdown("**W**ord **A**nalysis and **T**ext **T**racking with an **E**nhanced **L**anguage model **S**earch **E**ngine")
    st.markdown(
        "**W**holistic **A**nalysis of  **T**ex**T** with an **E**nhanced **L**anguage model **S**earch **E**ngine"
    )

    generator_config["custom_prompt"] = FR_USER_BASE_QUERY
    initialize_options_from_config()

    display_side_bar()

    display_existing_messages()

    query = st.chat_input("Enter any question in relation with the provided document")
    if query:
        add_user_message_to_session(query)

        response = generate_assistant_response(query)
        st.session_state["chat_history"].add_to_database(query, response)

    display_buttons()


if __name__ == "__main__":
    main()
