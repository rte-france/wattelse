from typing import List

from transformers import AutoTokenizer

from wattelse.llm.vllm_api import vLLM_API
from wattelse.summary.summarizer import (
    Summarizer,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_MAX_WORDS,
    DEFAULT_SUMMARIZATION_RATIO,
)
from wattelse.llm.prompts import FR_USER_SUMMARY_WORDS

TOKENIZER = AutoTokenizer.from_pretrained(
    "bofenghuang/vigogne-2-7b-chat",
    revision="v2.0",
    padding_side="right",
    use_fast=False,
)


class LocalLLMSummarizer(Summarizer):
    """Class that uses a local LLM service to provide a sumary of a text"""

    def __init__(self):
        self.api = vLLM_API()

    def generate_summary(
        self,
        article_text,
        max_sentences=DEFAULT_MAX_SENTENCES,
        max_words=DEFAULT_MAX_WORDS,
        max_length_ratio=DEFAULT_SUMMARIZATION_RATIO,
    ) -> str:
        # Generate a doc summary
        prompt = FR_USER_SUMMARY_WORDS.format(text=article_text, num_words=max_words)
        summary = self.api.generate(prompt)
        return summary

    def summarize_batch(
        self,
        article_texts: List[str],
        max_sentences: int=DEFAULT_MAX_SENTENCES,
        max_words=DEFAULT_MAX_WORDS,
        max_length_ratio: float=DEFAULT_SUMMARIZATION_RATIO,
    ) -> List[str]:
        return super().summarize_batch(article_texts, max_sentences, max_length_ratio)