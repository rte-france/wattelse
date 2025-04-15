#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import json
import uuid

from loguru import logger
from openai import OpenAI

from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_GET, require_POST


from wattelse.web_app.config.settings import CONFIG
from .models import Conversation, Message
from .utils import (
    streaming_generator_llm,
    conversation_messages,
    get_user_conversations,
)

LLM_CLIENT = OpenAI(base_url=CONFIG.gpt.base_url, api_key=CONFIG.gpt.api_key)

LLM_MAPPING = {
    "gpt-4o-mini": {
        "name": "GPT-4o mini",
        "provider": "openai",
    },
    "mistral-large-2411": {
        "name": "Mistral Large",
        "provider": "mistral",
    },
}

DEFAULT_MODEL_ID = "mistral-large-2411"


@login_required
@require_GET
def main_page(request):
    """
    Main function for GPT app.
    Renders GPT main page.
    """
    # Get user conversation history
    conversations = get_user_conversations(request.user)

    # Get available model names
    models = []
    for model in LLM_CLIENT.models.list().data:
        model_id = model.id
        # Add model to list if it is in the mapping
        if model_id in LLM_MAPPING:
            mapping = LLM_MAPPING[model_id]
            model_data = {
                "model_id": model_id,
                "model_name": mapping["name"],
                "provider": mapping["provider"],
            }
            models.append(model_data)
            if model_id == DEFAULT_MODEL_ID:
                default_model = model_data

    return render(
        request,
        "gpt/gpt.html",
        context={
            "conversations": conversations,
            "models": models,
            "default_model": default_model,
        },
    )


@login_required
@require_POST
def query_gpt(request):
    """
    Query LLM with user message.
    Saves user message to database.
    Returns LLM response as a stream.
    """
    # Get request data
    data = json.loads(request.body)
    conversation_id = uuid.UUID(data.get("conversation_id"))
    message_id = uuid.UUID(data.get("message_id"))
    content = data.get("content")
    model = data.get("model")

    print(model)

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
    messages = conversation_messages(
        conversation, n=CONFIG.gpt.max_messages, include_id=False
    )

    # Query LLM
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=CONFIG.gpt.max_tokens,
            stream=True,
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
    """
    Saves assistant message to database.
    This function is called after the streaming response from 'query_gpt' has ended.
    """
    # Get request data
    data = json.loads(request.body)
    conversation_id = uuid.UUID(data.get("conversation_id"))
    message_id = uuid.UUID(data.get("message_id"))
    content = data.get("content")
    model = data.get("model")

    # Get conversation
    conversation = Conversation.objects.get(id=conversation_id, user=request.user)

    # Add assistant message to database
    message = Message(
        id=message_id,
        conversation=conversation,
        role="assistant",
        content=content,
        model=model,
    )
    message.save()

    return HttpResponse(status=200)


@login_required
@require_GET
def get_conversation_messages(request):
    """
    Returns conversation messages associated with a conversation id in the OpenAI format.
    """
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
    """
    Handles user vote for a message.
    Retrieve message based on its id and update its rating.
    """
    # Get request data
    data = json.loads(request.body)
    message_id = data.get("message_id")
    rating = data.get("rating")

    # Get associated message and update rating
    message = Message.objects.get(id=message_id)
    message.rating = rating
    message.save()

    return HttpResponse(status=200)
