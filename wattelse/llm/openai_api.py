import configparser
from pathlib import Path

import openai
from loguru import logger


class OpenAI_API:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent.parent / "config" / "openai.cfg")
        openai.api_key = config.get("OPENAI_CONFIG", "openai_key")
        openai.organization = config.get("OPENAI_CONFIG", "openai_organization")

    def generate(
        self,
        user_prompt,
        system_prompt=None,
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=512,
    ) -> str:
        """Call openai model for generation.

            Args:
                    user_prompt (str): prompt to send to the model with role=user.
                    system_prompt (str): prompt to send to the model with role=system.
                    model_name (str, optional): name of the openai model to use for generation.
                    temperature (float, optional): Temperature for generation.
        max_tokens (int, optional): Maximum tokens to be generated.

            Returns:
                    str: model answer.
        """
        messages = [{"role": "user", "content": user_prompt}]
        # add system prompt if one is provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        try:
            answer = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.debug(f"API returned: {answer}")
            return answer.choices[0].message.content
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            msg = f"OpenAI API request timed out: {e}"
            logger.error(msg)
            return msg
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            msg = f"OpenAI API returned an API Error: {e}"
            logger.error(msg)
            return msg
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            msg = f"OpenAI API request failed to connect: {e}"
            logger.error(msg)
            return msg
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            msg = f"OpenAI API request was invalid: {e}"
            logger.error(msg)
            return msg
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            msg = f"OpenAI API request was not authorized: {e}"
            logger.error(msg)
            return msg
        except openai.error.PermissionError as e:
            # Handle permission error, e.g. check scope or log
            msg = f"OpenAI API request was not permitted: {e}"
            logger.error(msg)
            return msg
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            msg = f"OpenAI API request exceeded rate limit: {e}"
            logger.error(msg)
            return msg
