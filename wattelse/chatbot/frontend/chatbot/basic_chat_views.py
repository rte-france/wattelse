#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import configparser
from loguru import logger

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect

from openai import OpenAI

from wattelse.chatbot.backend import LLM_CONFIGS, FASTCHAT_LLM
from wattelse.chatbot.backend.rag_backend import RAGError
from wattelse.common.config_utils import parse_literal

from .utils import streaming_generator_llm

llm_config_file = LLM_CONFIGS.get(FASTCHAT_LLM, None)
if llm_config_file is None:
    raise RAGError(f"Unrecognized LLM API name")
config = configparser.ConfigParser(converters={"literal": parse_literal})
config.read(llm_config_file)
api_config = config["API_CONFIG"]
llm_config = {"api_key": api_config["openai_api_key"],
              "base_url": api_config["openai_url"],
              }
llm = OpenAI(**llm_config)


def basic_chat(request):
    """Main function for chatbot interface.
    If request method is GET : render chatbot.html
    If request method is POST : make a call to RAGOrchestratorClient and return response as json
    """
    if not request.user.is_authenticated:
        return redirect("/login")

    # If request method is POST, call RAG API and return response to query
    # else render template
    if request.method == "POST":
        message = request.POST.get("message", None)
        if not message:
            logger.warning(f"[User: {request.user.username}] No user message received")
            error_message = "Veuillez saisir une question"
            return JsonResponse({"error_message": error_message}, status=500)
        logger.info(f"[User: {request.user.username}] Query: {message}")

        # Query LLM
        try:
            response = llm.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            model=api_config["model_name"],
            stream=True,
            )
            logger.info("OK")
            return StreamingHttpResponse(streaming_generator_llm(response), status=200, content_type='text/event-stream')

        except Exception as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse({"error_message": f"Erreur lors de la requête au RAG: {e}"}, status=500)
    else:
        return render(
            request, "chatbot/llm.html",
        )
