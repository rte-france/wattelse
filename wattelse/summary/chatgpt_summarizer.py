import configparser
import openai
import tiktoken
from pathlib import Path
from loguru import logger

from wattelse.summary.summarizer import Summarizer

MODEL = "gpt-3.5-turbo-instruct"
BASE_PROMPT = "Résume en quelques phrases le texte suivant: "
MAX_TOKENS = 15
TEMPERATURE = 0


class GPTSummarizer(Summarizer):
    """Class that uses the GPT service to provide a sumary of a text"""

    def __init__(self):
        # retrieve chat GPT config
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent.parent / "config" / "openai.cfg")
        openai.api_key = config.get("OPENAI_CONFIG", "openai_key")
        openai.organization = config.get("OPENAI_CONFIG", "openai_organization")
        self.encoding = tiktoken.encoding_for_model(MODEL)

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        print(num_tokens)
        return num_tokens

    def generate_summary(self, article_text, max_length_ratio=0.1) -> str:
        answer = openai.Completion.create(
            model=MODEL,
            prompt=BASE_PROMPT + article_text,
            max_tokens=round(self.num_tokens_from_string(article_text)*max_length_ratio),
            temperature=TEMPERATURE,
        )
        logger.debug(f"API returned: {answer}")
        return answer.choices[0].text
