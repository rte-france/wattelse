import configparser
from typing import List

import tiktoken
from loguru import logger

from wattelse.llm.vars import MODEL, TEMPERATURE
from wattelse.summary.summarizer import (
    Summarizer,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_MAX_WORDS,
    DEFAULT_SUMMARIZATION_RATIO,
)
from wattelse.summary.summarizer import Summarizer
from wattelse.llm.prompts import FR_SYSTEM_SUMMARY_WORDS, EN_SYSTEM_SUMMARY_WORDS
from wattelse.llm.openai_api import OpenAI_API


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
        max_words=DEFAULT_MAX_WORDS,
        max_length_ratio: float=DEFAULT_SUMMARIZATION_RATIO,
        prompt_language = "fr",
        max_article_length=2000,
    ) -> str:
        # Limit input length :
        encoded_article_text = self.encoding.encode(article_text)
        if len(encoded_article_text) > max_article_length:
            encoded_article_text = encoded_article_text[0:max_article_length]
            article_text = self.encoding.decode(encoded_article_text)
        # Create answer object
        prompt = FR_SYSTEM_SUMMARY_WORDS if prompt_language=="fr" else EN_SYSTEM_SUMMARY_WORDS
        answer = self.api.generate(
            system_prompt=prompt.format(num_words=max_words),
            user_prompt=article_text,
            model_name=MODEL,
            temperature=TEMPERATURE,
        )
        return answer

    def summarize_batch(
        self,
        article_texts: List[str],
        max_sentences=DEFAULT_MAX_SENTENCES,
        max_words: int = DEFAULT_MAX_WORDS,
        max_length_ratio: float = DEFAULT_SUMMARIZATION_RATIO,
        prompt_language = "fr"
    ) -> List[str]:
        return super().summarize_batch(article_texts, max_sentences=max_sentences, max_words=max_words,
                                       max_length_ratio=max_length_ratio, prompt_language=prompt_language)
