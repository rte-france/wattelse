import configparser
from pathlib import Path
from typing import List

import numpy as np
import openai
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GenerationConfig

config = configparser.ConfigParser()
config.read(Path(__file__).parent.parent / "config" / "llm_api.cfg")
USE_REMOTE_LLM_MODEL = config.get("LLM_API_CONFIG", "use_remote_api")
OPENAI_KEY = config.get("LLM_API_CONFIG", "openai_key")
OPENAI_URL = config.get("LLM_API_CONFIG", "openai_url")

TEMPERATURE=0.1
MAX_TOKENS=512

BASE_PROMPT = """Répond à la question en utilisant le contexte fourni.
Contexte :
\"\"\"
{context}
\"\"\"
Question : {query}
La réponse doit être {expected_answer_size} et se baser uniquement sur le contexte. Si le contexte ne contient pas d'éléments permettant de répondre à la question, répondre \"Le contexte ne fourni pas assez d'information pour répondre à la question.\""""

def make_docs_embedding(docs: List[str], embedding_model: SentenceTransformer):
    return embedding_model.encode(docs, show_progress_bar=True)


def extract_n_most_relevant_extracts(n, query, docs, docs_embeddings, embedding_model, similarity_threshold:float = 0):
    query_embedding = embedding_model.encode(query)
    similarity = cosine_similarity([query_embedding], docs_embeddings)[0]

    # Find indices of documents with similarity above threshold
    above_threshold_indices = np.where(similarity > similarity_threshold)[0]

    # Sort above-threshold indices by similarity and select top n
    max_indices = above_threshold_indices[np.argsort(similarity[above_threshold_indices])][-n:][::-1]

    return docs[max_indices].tolist(), similarity[max_indices].tolist()

def generate_RAG_prompt(query: str, context_elements: List[str], expected_answer_size="short", custom_prompt=None) -> str:
    """
    Generates RAG prompt using query and context.
    """
    context = "\n".join(context_elements)
    expected_answer_size = "courte" if expected_answer_size=="short" else "détaillée"
    if custom_prompt:
        return custom_prompt.format(context=context, query=query, expected_answer_size=expected_answer_size)
    else:
        return BASE_PROMPT.format(context=context, query=query, expected_answer_size=expected_answer_size)

def generate_llm_specific_prompt(prompt, tokenizer):
    """
    Takes a prompt as input and returns the prompt in the LLM specific format.
    """
    chat = [{
            "role": "user",
            "content": prompt,
            }]
    return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize = False)


def generate_answer_locally(instruct_model, tokenizer, prompt) -> str:
    """Uses the local model to generate the answer"""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
        instruct_model.device
    )
    input_length = input_ids.shape[1]
    generated_outputs = instruct_model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=TEMPERATURE,
            do_sample=True,
            repetition_penalty=1.0,
            max_new_tokens=MAX_TOKENS,
        ),
        return_dict_in_generate=True,
    )
    generated_tokens = generated_outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logger.debug(f"Answer: {generated_text}")
    return generated_text

def generate_answer_remotely(prompt) -> str:
    """
    Uses the remote model (API) to generate the answer. Return full text once generated.
    """

    logger.debug(f"Calling remote LLM service...")
    print(prompt)
    try:
        model = get_api_model_name()

        # Use of completion API
        completion_result = openai.api_resources.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        answer = completion_result["choices"][0]["text"]
        logger.debug(f"Answer: {answer}")
        return answer
    except Exception as e:
        msg = f"Cannot reach the API endpoint: {OPENAI_URL}. Error: {e}"
        logger.error(msg)
        return msg

def generate_answer_remotely_stream(prompt) -> str:
    """
    Uses the remote model (API) to generate the answer. Returns Reponse object streaming output.
    """

    logger.debug(f"Calling remote LLM service...")
    print(prompt)
    try:
        model = get_api_model_name()

        # Use of completion API
        completion_result = openai.api_resources.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )
        return completion_result
    except Exception as e:
        msg = f"Cannot reach the API endpoint: {OPENAI_URL}. Error: {e}"
        logger.error(msg)
        return msg

def get_api_model_name():
    """
    Return currently loaded llm (using vLLM).
    """
    try :
        openai.api_key = OPENAI_KEY
        openai.api_base = OPENAI_URL

        # First model
        models = openai.Model.list()
        return models["data"][0]["id"]
    except Exception as e:
        return "API is down"