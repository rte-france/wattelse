#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
import configparser
from loguru import logger

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect

from openai import OpenAI

from wattelse.chatbot.backend import LLM_CONFIGS, FASTCHAT_LLM
from wattelse.chatbot.backend.rag_backend import RAGError
from wattelse.common.config_utils import parse_literal

from .utils import streaming_generator_llm

# Load locally deployed LLM configuration file
llm_config_file = LLM_CONFIGS.get(FASTCHAT_LLM, None)
if llm_config_file is None:
    raise RAGError(f"Unrecognized LLM API name")
config = configparser.ConfigParser(converters={"literal": parse_literal})
config.read(llm_config_file)

# Initialize OpenAI API with configuration parameters from config file
api_config = config["API_CONFIG"]
llm_config = {"api_key": api_config["openai_api_key"],
              "base_url": api_config["openai_url"],
              }
LLM_CLIENT = OpenAI(**llm_config)


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
        # Get conversation history
        history = json.loads(request.body)
        if not history:
            logger.warning(f"[User: {request.user.username}] No user message received")
            error_message = "Veuillez saisir une question"
            return JsonResponse({"error_message": error_message}, status=500)
        logger.info(f"[User: {request.user.username}] Message: {history[-1]['content']}")

        # Query LLM
        try:
            response = LLM_CLIENT.chat.completions.create(
                messages=history,
                model=api_config["model_name"],
                stream=True,
            )
            return StreamingHttpResponse(streaming_generator_llm(response), status=200, content_type='text/event-stream')

        except Exception as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse({"error_message": f"Erreur lors de la requête au RAG: {e}"}, status=500)
    else:
        return render(request, "chatbot/secureGPT.html", context={"llm_name": api_config["model_name"]})
