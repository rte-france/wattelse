import configparser
from pathlib import Path
from loguru import logger

import openai
from loguru import logger


config = configparser.ConfigParser()
config.read(Path(__file__).parent.parent / "config" / "openai.cfg")
openai.api_key = config.get("OPENAI_CONFIG", "openai_key")
openai.organization = config.get("OPENAI_CONFIG", "openai_organization")


class OpenAI_API():

	def __init__(self):
		pass

	def generate(self, prompt, model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=512) -> str:
		"""Call openai model for generation.

		Args:
			prompt (str): prompt to send to the model.
			model_name (str, optional): name of the openai model to use for generation.
			temperature (float, optional): Temperature for generation.
            max_tokens (int, optional): Maximum tokens to be generated.

		Returns:
			str: model answer.
		"""
		answer = openai.ChatCompletion.create(
				model=model_name,
				messages=[{"role": "user", "content": prompt}],
				max_tokens=max_tokens,
				temperature=temperature,
			)
		logger.debug(f"API returned: {answer}")
		return answer.choices[0].message.content




