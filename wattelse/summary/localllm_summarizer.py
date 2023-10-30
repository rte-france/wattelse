from transformers import AutoTokenizer
from vigogne.preprocess import generate_inference_chat_prompt

from wattelse.chatbot.utils import generate_answer_remotely
from wattelse.summary.summarizer import Summarizer
from wattelse.summary.prompts import BASE_PROMPT_SUMMARY

TOKENIZER = AutoTokenizer.from_pretrained("bofenghuang/vigogne-2-7b-chat", revision="v2.0", padding_side="right", use_fast=False)

class LocalLLMSummarizer(Summarizer):
    """Class that uses a local LLM service to provide a sumary of a text"""

    def __init__(self):
        pass

    def generate_summary(self, article_text, max_length_ratio=0.1) -> str:
        # Generate a doc summary
        prompt = BASE_PROMPT_SUMMARY + article_text
        prompt = generate_inference_chat_prompt([[prompt,""]], TOKENIZER, max_length = 4096)
        summary = generate_answer_remotely(prompt)
        return summary