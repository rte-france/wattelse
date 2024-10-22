#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import configparser
import json
import os
import uuid
import datetime
from pathlib import Path

from loguru import logger

from django.http import JsonResponse, StreamingHttpResponse, Http404
from django.shortcuts import render, redirect

from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.common.config_utils import parse_literal, EnvInterpolation
from .models import GPTChat
from .utils import (
    streaming_generator_llm,
    get_conversation_history,
    get_user_conversation_ids,
    get_conversation_first_message,
)

# NUMBER MAX OF TOKENS
MAX_TOKENS = 1536


# Config for retriever and generator
config = configparser.ConfigParser(
    converters={"literal": parse_literal}, interpolation=EnvInterpolation()
)  # takes into account environment variables
config.read(Path(__file__).parent / "secure_gpt.cfg")
openai_cfg = parse_literal(dict(config["openai_cfg"]))

llm_config = {
    "api_key": openai_cfg["openai_api_key"],
    "endpoint": openai_cfg["openai_endpoint"],
    "model": openai_cfg["openai_default_model"],
    "temperature": openai_cfg["temperature"],
}

LLM_CLIENT = OpenAI_Client(**llm_config)

LLM_MAPPING = {
    "wattelse-gpt35": "gpt-3.5",
    "wattelse-gpt4": "gpt-4",
    "wattelse-gpt4o-mini-sweden": "gpt4o-mini",
    "wattelse-gpt4o-sweden": "gpt4o",
}


def request_client(request):
    """Main function for chatbot interface.
    If request method is GET : render chatbot.html
    If request method is POST : make a call to OpenAI client
    """
    # If user is not authenticated, redirect to login page
    if not request.user.is_authenticated:
        return redirect("/login")

    # If request method is POST, call OpenAI client
    # Else render template
    if request.method == "POST":
        # Get request data
        data = json.loads(request.body)

        # Get conversation id
        conversation_id = uuid.UUID(data.get("conversation_id"))

        # Get user chat history
        history = get_conversation_history(
            request.user, conversation_id, ChatModel=GPTChat
        )

        # Get posted message
        user_message = data.get("message", None)

        # Check message is not empty
        if not user_message:
            logger.warning(f"[User: {request.user.username}] No user message received")
            return JsonResponse({"message": "Aucune question reçue"}, status=500)

        # Log message
        logger.info(f"[User: {request.user.username}] Query: {user_message}")

        if not history:
            history = []
        messages = history + [{"role": "user", "content": user_message}]

        # Query LLM
        try:
            response = LLM_CLIENT.generate_from_history(
                messages=messages,
                stream=True,
                max_tokens=MAX_TOKENS,
            )
            return StreamingHttpResponse(
                streaming_generator_llm(response),
                status=200,
                content_type="text/event-stream",
            )

        except Exception as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse(
                {"error_message": f"Erreur lors de la requête au RAG: {e}"}, status=500
            )
    else:
        # Get conversation ids
        conversation_ids = get_user_conversation_ids(
            request.user,
            ChatModel=GPTChat,
        )

        # Sort conversations into date categories for better user expericence
        sorted_conversations = {"today": [], "last_week": [], "others": []}

        today = datetime.datetime.now()
        for id in conversation_ids:
            # Get conversation title and timestamp
            title, timestamp = get_conversation_first_message(id, GPTChat)

            # If created today
            if timestamp.date() == today.date():
                sorted_conversations["today"].append({"id": id, "title": title})

            # If created in 7 last days
            elif timestamp.date() >= (today - datetime.timedelta(days=7)).date():
                sorted_conversations["last_week"].append({"id": id, "title": title})

            # If created longer ago
            else:
                sorted_conversations["others"].append({"id": id, "title": title})

        print(llm_config["model"][1:])
        return render(
            request,
            "chatbot/secureGPT.html",
            context={
                "llm_name": LLM_MAPPING[llm_config["model"]],
                "conversations": sorted_conversations,
            },
        )


def get_conversation_messages(request):
    """
    Function to send the list of messages of a conversation to the frontend.
    """
    if request.method == "POST":
        # Get response data
        data = json.loads(request.body)
        conversation_id = uuid.UUID(data.get("id"))

        # Get conversation messages
        try:
            history = get_conversation_history(
                request.user, conversation_id, ChatModel=GPTChat
            )
            # Return JsonReponse with success message that will be shown in frontend
            return JsonResponse(
                {"message": f"Modifications validées", "history": history}
            )

        # Return error message if something goes wrong
        except Exception as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse(
                {
                    "message": "Erreur serveur : échec lors de la récupération de la conversation"
                },
                status=500,
            )
    else:
        raise Http404()
