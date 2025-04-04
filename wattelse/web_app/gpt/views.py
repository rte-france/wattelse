#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import configparser
import json
import uuid
from pathlib import Path

from loguru import logger

from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_GET, require_POST

from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.config_utils import parse_literal, EnvInterpolation
from .models import Conversation, Message
from .utils import (
    streaming_generator_llm,
    conversation_messages,
    get_user_conversations,
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
@require_GET
def main_page(request):
    """Main function for GPT app.
    If request method is GET : render gpt.html
    If request method is POST : make a call to OpenAI client
    """
    # Get user conversation history
    conversations = get_user_conversations(request.user)

    return render(
        request,
        "gpt/gpt.html",
        context={
            "conversations": conversations,
        },
    )


@login_required
@require_POST
def query_gpt(request):
    # Get request data
    data = json.loads(request.body)
    conversation_id = uuid.UUID(data.get("conversation_id"))
    message_id = uuid.UUID(data.get("message_id"))
    content = data.get("content")

    # Get or create conversation
    conversation, _ = Conversation.objects.get_or_create(
        id=conversation_id,
        user=request.user,
        defaults={"title": content},
    )

    # Add user message to database
    message = Message(
        id=message_id, conversation=conversation, role="user", content=content
    )
    message.save()

    # Get user chat history
    history = conversation_messages(conversation, n=MAX_MESSAGES)

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
        return JsonResponse({"detail": f"Erreur lors de l'appel au LLM"}, status=500)


@login_required
@require_POST
def save_assistant_message(request):
    # Get request data
    data = json.loads(request.body)
    conversation_id = uuid.UUID(data.get("conversation_id"))
    message_id = uuid.UUID(data.get("message_id"))
    content = data.get("content")

    # Get conversation
    conversation = Conversation.objects.get(id=conversation_id, user=request.user)

    # Add assistant message to database
    message = Message(
        id=message_id,
        conversation=conversation,
        role="assistant",
        content=content,
    )
    message.save()

    return HttpResponse(status=200)


@login_required
@require_GET
def get_conversation_messages(request):
    # Get request data
    conversation_id = uuid.UUID(request.GET.get("id"))

    # Get conversation
    conversation = Conversation.objects.get(id=conversation_id, user=request.user)

    # Get conversation history
    messages = conversation_messages(conversation)

    return JsonResponse({"messages": messages}, status=200)


@login_required
@require_POST
def handle_vote(request):
    # Get request data
    data = json.loads(request.body)
    message_id = data.get("message_id")
    rating = data.get("rating")

    # Get associated message and update rating
    message = Message.objects.get(id=message_id)
    message.rating = rating
    message.save()

    return HttpResponse(status=200)
