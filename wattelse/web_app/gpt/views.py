#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import configparser
import json
import uuid
from pathlib import Path

from loguru import logger

from django.http import HttpResponse, JsonResponse, StreamingHttpResponse, Http404
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.config_utils import parse_literal, EnvInterpolation
from .models import Conversation, Message
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


@login_required
def main_page(request):
    """Main function for GPT app.
    If request method is GET : render gpt.html
    If request method is POST : make a call to OpenAI client
    """
    if request.method == "GET":
        # Get user conversation history
        conversations = get_user_conversation_history(request.user)

        return render(
            request,
            "gpt/secureGPT.html",
            context={
                "llm_name": LLM_MAPPING[llm_config["model"]],
                "conversations": conversations,
            },
        )
    else:
        raise Http404()


@login_required
def query_gpt(request):
    if request.method == "POST":
        # Get request data
        data = json.loads(request.body)
        conversation_id = uuid.UUID(data.get("conversation_id"))
        content = data.get("content")

        # Add user message to database
        add_message_to_db(conversation_id, role="user", content=content)

        # Get user chat history
        history = get_conversation_history(
            request.user, conversation_id, n=MAX_MESSAGES
        )

        # Query LLM
        try:
            response = LLM_CLIENT.generate_from_history(
                messages=history,
                stream=True,
                max_tokens=MAX_TOKENS,
            )
            return StreamingHttpResponse(
                streaming_generator_llm(response),
                status=200,
                content_type="text/event-stream",
            )

        except Exception as e:
            logger.error(e)
            return JsonResponse(
                {"detail": f"Erreur lors de l'appel au LLM"}, status=500
            )
    else:
        raise Http404()


def add_message_to_db(conversation_id: uuid.UUID, role: str, content: str) -> None:
    """Saves a message to Django database using model Message"""
    message = Message(conversation=conversation_id, role=role, content=content)
    message.save()
