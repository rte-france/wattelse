import configparser
import openai
import tiktoken
from pathlib import Path
from loguru import logger
from openai import APIError

from wattelse.summary.summarizer import Summarizer
from wattelse.llm.prompts import BASE_PROMPT_SUMMARY

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.1


class GPTSummarizer(Summarizer):
    """Class that uses the GPT service to provide a sumary of a text"""

    def __init__(self):
        # retrieve chat GPT config
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent.parent / "config" / "openai.cfg")
        openai.api_key = config.get("OPENAI_CONFIG", "openai_key")
        openai.organization = config.get("OPENAI_CONFIG", "openai_organization")
        self.encoding = tiktoken.encoding_for_model(MODEL)
        logger.debug("GPTSummarizer initialized")

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def generate_summary(self, article_text, max_summary_length_ratio=0.5, max_article_length=2000) -> str:
        try:
            # Limit input length :
            encoded_article_text = self.encoding.encode(article_text)
            if len(encoded_article_text) > max_article_length:
                encoded_article_text = encoded_article_text[0:max_article_length]
                article_text = self.encoding.decode(encoded_article_text)
            # Create answer object
            answer = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": BASE_PROMPT_SUMMARY.format(text=article_text)}],
                max_tokens=round(self.num_tokens_from_string(article_text)*max_summary_length_ratio),
                temperature=TEMPERATURE,
            )
            logger.debug(f"API returned: {answer}")
            return answer.choices[0].message.content
        except APIError as e:
            return f"OpenAI API error : {e}"
            logger.error(f"OpenAI API error : {e}")
