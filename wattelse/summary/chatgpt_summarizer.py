import configparser
from typing import List

import openai
import tiktoken
from pathlib import Path
from loguru import logger
from openai import APIError

from wattelse.summary.summarizer import (
    Summarizer,
    DEFAULT_SUMMARIZATION_RATIO,
    DEFAULT_MAX_SENTENCES,
)
from wattelse.summary.summarizer import Summarizer
from wattelse.llm.prompts import FR_SYSTEM_SUMMARY
from wattelse.llm.openai_api import OpenAI_API

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.1


class GPTSummarizer(Summarizer):
    """Class that uses the GPT service to provide a sumary of a text"""

    def __init__(self):
        # retrieve chat GPT config
        self.api = OpenAI_API()
        self.encoding = tiktoken.encoding_for_model(MODEL)
        logger.debug("GPTSummarizer initialized")

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def generate_summary(
        self,
        article_text,
        max_sentences=DEFAULT_MAX_SENTENCES,
        max_summary_length_ratio=DEFAULT_SUMMARIZATION_RATIO,
        max_article_length=2000,
    ) -> str:
        # Limit input length :
        encoded_article_text = self.encoding.encode(article_text)
        if len(encoded_article_text) > max_article_length:
            encoded_article_text = encoded_article_text[0:max_article_length]
            article_text = self.encoding.decode(encoded_article_text)
        # Create answer object
        answer = self.api.generate(
            system_prompt = FR_SYSTEM_SUMMARY.format(num_sentences=max_sentences),
            user_prompt = article_text,
            model_name = MODEL,
            temperature = TEMPERATURE,
            max_tokens=round(self.num_tokens_from_string(article_text)*max_summary_length_ratio),
        )
        return answer

    def summarize_batch(
        self,
        article_texts: List[str],
        max_sentences: int=DEFAULT_MAX_SENTENCES,
        max_length_ratio: float=DEFAULT_SUMMARIZATION_RATIO,
    ) -> List[str]:
        return super().summarize_batch(article_texts, max_sentences, max_length_ratio)