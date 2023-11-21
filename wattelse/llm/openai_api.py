import configparser
from pathlib import Path

import openai
from loguru import logger

MAX_ATTEMPTS = 3


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
        current_attempt=1,
    ) -> str:
        """Call openai model for generation.

            Args:
                    user_prompt (str): prompt to send to the model with role=user.
                    system_prompt (str): prompt to send to the model with role=system.
                    model_name (str, optional): name of the openai model to use for generation.
                    temperature (float, optional): Temperature for generation.
        max_tokens (int, optional): Maximum tokens to be generated.
                    current_try: id of current try (in case of failure, this is increased and another try is done)

            Returns:
                    str: model answer.
        """
        if current_attempt >= MAX_ATTEMPTS:
            logger.error(
                "Maximum number of API call attempts reached. Request cancelled. Check previous logs for details."
            )
            return (
                f"OpenAI API fatal error: Maximum number of API call attempts reached."
            )

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
        # Details of errors available here: https://platform.openai.com/docs/guides/error-codes/api-errors
        except (
            openai.error.APIConnectionError,
            openai.error.APIError,
            openai.error.AuthenticationError,
            openai.error.InvalidAPIType,
            openai.error.InvalidRequestError,
            openai.error.PermissionError,
            openai.error.SignatureVerificationError,
        ) as e:
            # Fatal errors, do not retry
            msg = f"OpenAI API fatal error: {e}"
            logger.error(msg)
            return msg
        except (
            openai.error.APIError,
            openai.error.RateLimitError,
            openai.error.Timeout,
            openai.error.TryAgain,
            openai.error.ServiceUnavailableError,
        ) as e:
            # Non-fatal errors, handle retry request
            logger.error(f"OpenAI API non-fatal error: {e}")
            logger.warning(f"Retrying the same request...")
            return self.generate(
                user_prompt,
                system_prompt,
                model_name,
                temperature,
                max_tokens,
                current_attempt + 1,
            )
