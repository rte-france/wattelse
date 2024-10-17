#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
import os
import uuid

from loguru import logger

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect

from openai import OpenAI, AzureOpenAI, Timeout

from wattelse.api.openai.client_openai_api import (
    AZURE_API_VERSION,
    MAX_ATTEMPTS,
    TIMEOUT,
)
from .utils import streaming_generator_llm, get_conversation_history

# Uses environment variables to configure the openai API
# TODO: add support for Azure
var_prefix = "AZURE_SE_WATTELSE_"
api_key = os.getenv(f"{var_prefix}OPENAI_API_KEY")
if not api_key:
    logger.error(
        f"WARNING: {var_prefix}OPENAI_API_KEY environment variable not found. Please set it before using OpenAI services."
    )
    raise EnvironmentError(
        f"{var_prefix}OPENAI_API_KEY environment variable not found."
    )
endpoint = os.getenv(f"{var_prefix}OPENAI_ENDPOINT", None)
if endpoint == "":  # check empty env var
    endpoint = None
run_on_azure = "azure.com" in endpoint if endpoint else False
model_name = os.getenv(f"{var_prefix}OPENAI_DEFAULT_MODEL_NAME", None)

common_params = {
    "api_key": api_key,
    "timeout": Timeout(TIMEOUT, connect=10.0),
    "max_retries": MAX_ATTEMPTS,
}
openai_params = {
    "base_url": endpoint,
}
azure_params = {"azure_endpoint": endpoint, "api_version": AZURE_API_VERSION}

if not run_on_azure:
    LLM_CLIENT = OpenAI(
        **common_params,
        **openai_params,
    )
else:
    LLM_CLIENT = AzureOpenAI(
        **common_params,
        **azure_params,
    )


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
        history = get_conversation_history(request.user, conversation_id)

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
            response = LLM_CLIENT.chat.completions.create(
                messages=messages,
                model=model_name,
                stream=True,
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
        return render(
            request, "chatbot/secureGPT.html", context={"llm_name": model_name}
        )
