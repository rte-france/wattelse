import configparser

from django.http import JsonResponse
from django.shortcuts import render, redirect
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from wattelse.chatbot.backend import LLM_CONFIGS, FASTCHAT_LLM
from wattelse.chatbot.backend.rag_backend import RAGError
from wattelse.common.config_utils import parse_literal

llm_config_file = LLM_CONFIGS.get(FASTCHAT_LLM, None)
if llm_config_file is None:
    raise RAGError(f"Unrecognized LLM API name")
config = configparser.ConfigParser(converters={"literal": parse_literal})
config.read(llm_config_file)
api_config = config["API_CONFIG"]
llm_config = {"openai_api_key": api_config["openai_api_key"],
              "openai_api_base": api_config["openai_url"],
              "model_name": api_config["model_name"],
              "temperature": api_config["temperature"],
              }
llm = ChatOpenAI(**llm_config)


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
            logger.warning("No user message received")
            error_message = "Veuillez saisir une question"
            return JsonResponse({"error_message": error_message}, status=500)
        logger.info(f"User: {request.user.username} - Query: {message}")

        # Query LLM
        template = """Question: {question}

        Réponse: Let's think step by step."""
        prompt = PromptTemplate.from_template(template)
        try:
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            answer = llm_chain.run(message)
            return JsonResponse({"message": message, "answer": answer}, status=200)

        except Exception as e:
            logger.error(e)
            return JsonResponse({"error_message": f"Erreur lors de la requête au RAG: {e}"}, status=500)
    else:
        return render(
            request, "chatbot/llm.html",
        )
