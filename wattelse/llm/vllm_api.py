import configparser
from pathlib import Path
from typing import List

import openai
from loguru import logger
from transformers import AutoTokenizer


def get_api_model_name():
    """
    Return currently loaded model name in vLLM.
    """
    try:
        # First model
        models = openai.Model.list()
        return models["data"][0]["id"]
    except Exception as e:
        return None


class vLLM_API:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent.parent / "config" / "llm_api.cfg")
        openai.api_key = config.get("LLM_API_CONFIG", "openai_key")
        openai.api_base = config.get("LLM_API_CONFIG", "openai_url")

        self.model_name = get_api_model_name()
        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name, padding_side="right", use_fast=False
            )
            if self.model_name
            else None
        )

    def generate_llm_specific_prompt(self, user_prompt, system_prompt=None) -> str:
        """
        Takes a prompt as input and returns the prompt in the specific LLM format.
        """
        messages = [{"role": "user", "content": user_prompt}]
        # add system prompt if one is provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def generate(
        self,
        user_prompt,
        system_prompt=None,
        temperature=0.1,
        max_tokens=512,
        transform_prompt=True,
        stream=False,
    ) -> str:
        """Uses the remote model (API) to generate the answer.

        Args:
            user_prompt (str): prompt to send to the model with role=user.
            system_prompt (str): prompt to send to the model with role=system. Useless if transform_prompt=False.
            temperature (float, optional): Temperature for generation.
            max_tokens (int, optional): Maximum tokens to be generated.
            transform_prompt (bool, optional): If True, transforms the input prompt to match the chat template used in LLM training.
            stream (bool, optional): Whether to stream output. Defaults to False.

        Returns:
            str: Returns :
                    - full text once generated if stream=False
                    - Completion object if stream=True
        """
        logger.debug(f"Calling remote vLLM service...")
        try:
            if transform_prompt:
                prompt = self.generate_llm_specific_prompt(user_prompt, system_prompt=system_prompt)

            # Use of completion API
            completion_result = openai.api_resources.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
            if stream:
                return completion_result
            else:
                return completion_result["choices"][0]["text"]

        except Exception as e:
            msg = f"Cannot reach the API endpoint: {openai.api_base}. Error: {e}"
            logger.error(msg)
            return msg
