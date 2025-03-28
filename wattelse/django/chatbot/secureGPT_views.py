#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import configparser
import json
import uuid
from pathlib import Path

from loguru import logger

from django.http import JsonResponse, StreamingHttpResponse, Http404
from django.shortcuts import render, redirect

from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.config.config_utils import parse_literal, EnvInterpolation
from .models import GPTChat
from .utils import (
    streaming_generator_llm,
    get_conversation_history,
    get_user_conversation_history,
    LLM_MAPPING,
)

# NUMBER MAX OF TOKENS
MAX_TOKENS = 1536

# Max messages in history
MAX_MESSAGES = 12


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


def gpt_page(request):
    """Main function for chatbot interface.
    If request method is GET : render chatbot.html
    If request method is POST : make a call to OpenAI client
    """
    # If user is not authenticated, redirect to login page
    if not request.user.is_authenticated:
        return redirect("/login")

    if request.method == "GET":
        # Get user conversation history
        conversations = get_user_conversation_history(request.user, GPTChat)

        return render(
            request,
            "chatbot/secureGPT.html",
            context={
                "llm_name": LLM_MAPPING[llm_config["model"]],
                "conversations": conversations,
            },
        )
    else:
        raise Http404()


def query_gpt(request):
    if request.method == "POST":
        # Get request data
        data = json.loads(request.body)

        # Get conversation id
        conversation_id = uuid.UUID(data.get("conversation_id"))

        # Get user chat history
        history = get_conversation_history(
            request.user, conversation_id, ChatModel=GPTChat, n=MAX_MESSAGES
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
        raise Http404()
