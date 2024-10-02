#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import os

from openai import OpenAI, AzureOpenAI, Timeout, Stream
from loguru import logger
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion, ChatCompletionChunk

MAX_ATTEMPTS = 3
TIMEOUT = 60.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512

AZURE_API_VERSION = "2024-02-01"


class OpenAI_Client:
    """Generic client for Open AI API (either direct API or via Azure).
    Important note: the API key and the ENDPOINT must be set using environment variables OPENAI_API_KEY and
    OPENAI_ENDPOINT respectively. (The endpoint shall only be set for Azure or local deployment)
    """

    def __init__(self, api_key: str = None, endpoint: str = None):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "WARNING: OPENAI_API_KEY environment variable not found. Please set it before using OpenAI services."
            )
            raise EnvironmentError(f"OPENAI_API_KEY environment variable not found.")

        endpoint = endpoint if endpoint else os.getenv("OPENAI_ENDPOINT", None)
        if endpoint == "":  # check empty env var
            endpoint = None

        run_on_azure = "azure.com" in endpoint if endpoint else False

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
            self.llm_client = OpenAI(
                **common_params,
                **openai_params,
            )
        else:
            self.llm_client = AzureOpenAI(
                **common_params,
                **azure_params,
            )
        self.model_name = os.getenv("OPENAI_DEFAULT_MODEL_NAME")
        self.temperature = DEFAULT_TEMPERATURE
        self.max_tokens = DEFAULT_MAX_TOKENS

    def generate(
        self,
        user_prompt,
        system_prompt=None,
        **kwargs,
    ) -> ChatCompletion | Stream[ChatCompletionChunk] | str:
        """Call openai model for generation.

        Args:
                user_prompt (str): prompt to send to the model with role=user.
                system_prompt (str): prompt to send to the model with role=system.
                model_name (str, optional): name of the openai model to use for generation.
                temperature (float, optional): Temperature for generation.
                max_tokens (int, optional): Maximum tokens to be generated.
                seed: seed for generation
                stream: indicated if the result has to be streamed or not

        Returns:
                (str or Stream[ChatCompletionChunk]): model answer.

        """
        # Transform messages into OpenAI API compatible format
        messages = [{"role": "user", "content": user_prompt}]
        # Add system prompt if one is provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # For important parameters, set default value if not given
        model = kwargs.get("model", self.model_name)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            answer = self.llm_client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            logger.debug(f"API returned: {answer}")
            if kwargs.get("stream", False):
                return answer
            else:
                return answer.choices[0].message.content
        # Details of errors available here: https://platform.openai.com/docs/guides/error-codes/api-errors
        except Exception as e:
            msg = f"OpenAI API fatal error: {e}"
            logger.error(msg)
            return msg
