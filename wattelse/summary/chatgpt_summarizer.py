#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from typing import List

import tiktoken
from loguru import logger

from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.api.prompts import FR_SYSTEM_SUMMARY_WORDS, EN_SYSTEM_SUMMARY_WORDS
from wattelse.summary.summarizer import (
    DEFAULT_MAX_SENTENCES,
    DEFAULT_MAX_WORDS,
    DEFAULT_SUMMARIZATION_RATIO,
)
from wattelse.summary.summarizer import Summarizer


class GPTSummarizer(Summarizer):
    """Class that uses the GPT service to provide a summary of a text"""

    def __init__(self, api_key: str = None, endpoint: str = None):
        # retrieve chat GPT config
        self.api = OpenAI_Client(api_key=api_key, endpoint=endpoint)
        logger.debug("GPTSummarizer initialized")

    def generate_summary(
            self,
            article_text: str,
            max_sentences: int = DEFAULT_MAX_SENTENCES,
            max_words: int = DEFAULT_MAX_WORDS,
            max_length_ratio: float = DEFAULT_SUMMARIZATION_RATIO,
            prompt_language: str = "fr",
            max_article_length: int = 1500,
            model_name: str = None,
    ) -> str:
        # Limit input length in case the text is large
        article_text = keep_first_n_words(article_text, max_article_length)

        # Create answer object
        prompt = (
            FR_SYSTEM_SUMMARY_WORDS
            if prompt_language == "fr"
            else EN_SYSTEM_SUMMARY_WORDS
        )
        answer = self.api.generate(
            system_prompt=prompt.format(num_words=max_words),
            user_prompt=article_text,
        )
        return answer


def keep_first_n_words(text: str, n: int) -> str:
    """This function keeps the first n words of a text.
  Args:
      text: The text string.
      n: The number of words to keep.
  Returns:
      A string containing the first n words of the text.
  """
    words = text.split()
    if n > len(words):
        return text  # Handle case where n is larger than the number of words
    return " ".join(words[:n])
