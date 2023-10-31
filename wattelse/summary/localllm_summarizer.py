from transformers import AutoTokenizer

from wattelse.llm.vllm_api import vLLM_API
from wattelse.summary.summarizer import Summarizer
from wattelse.llm.prompts import BASE_PROMPT_SUMMARY

TOKENIZER = AutoTokenizer.from_pretrained("bofenghuang/vigogne-2-7b-chat", revision="v2.0", padding_side="right", use_fast=False)

class LocalLLMSummarizer(Summarizer):
    """Class that uses a local LLM service to provide a sumary of a text"""

    def __init__(self):
        self.api = vLLM_API()

    def generate_summary(self, article_text, max_length_ratio=0.1) -> str:
        # Generate a doc summary
        prompt = BASE_PROMPT_SUMMARY.format(text=article_text)
        summary = self.api.generate(prompt)
        return summary