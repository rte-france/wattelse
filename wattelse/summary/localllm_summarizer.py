#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
from pathlib import Path
from typing import List

from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.summary.summarizer import (
    Summarizer,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_MAX_WORDS,
    DEFAULT_SUMMARIZATION_RATIO,
)
from wattelse.api.prompts import FR_USER_SUMMARY_WORDS, EN_USER_SUMMARY_WORDS

class LocalLLMSummarizer(Summarizer):
    """Class that uses Fastchat LLM service to provide a sumary of a text"""

    def __init__(self):
        self.api = OpenAI_Client(config_path=Path(__file__).parent.parent / "api" / "openai" / "local_openai.cfg")

    def generate_summary(
        self,
        article_text,
        max_sentences=DEFAULT_MAX_SENTENCES,
        max_words=DEFAULT_MAX_WORDS,
        max_length_ratio=DEFAULT_SUMMARIZATION_RATIO,
        prompt_language="fr"
    ) -> str:
        # Generate a doc summary
        prompt = (FR_USER_SUMMARY_WORDS if prompt_language=="fr" else EN_USER_SUMMARY_WORDS).format(text=article_text, num_words=max_words)
        summary = self.api.generate(prompt)
        return summary

    def summarize_batch(
        self,
        article_texts: List[str],
        max_sentences: int=DEFAULT_MAX_SENTENCES,
        max_words=DEFAULT_MAX_WORDS,
        max_length_ratio: float=DEFAULT_SUMMARIZATION_RATIO,
        prompt_language="fr"
    ) -> List[str]:
        return super().summarize_batch(article_texts, max_sentences, max_words, max_length_ratio, prompt_language)