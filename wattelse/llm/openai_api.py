import configparser
from pathlib import Path

import openai
from openai import OpenAI, Timeout
from loguru import logger
from openai._types import NOT_GIVEN

MAX_ATTEMPTS = 3
TIMEOUT = 60.0

class OpenAI_API:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent.parent / "config" / "openai.cfg")
        self.llm_client = OpenAI(
            api_key=config.get("OPENAI_CONFIG", "openai_key"),
            organization=config.get("OPENAI_CONFIG", "openai_organization"),
            timeout=Timeout(TIMEOUT, connect=10.0),
            max_retries=MAX_ATTEMPTS,
        )
        self.model_name = "openai_gpt"

    def generate(
        self,
        user_prompt,
        system_prompt=None,
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=512,
        seed=NOT_GIVEN,
        stream=NOT_GIVEN,
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
        messages = [{"role": "user", "content": user_prompt}]
        # add system prompt if one is provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        try:
            answer = self.llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            logger.debug(f"API returned: {answer}")
            if stream:
                return answer
            else:
                return answer.choices[0].message.content
        # Details of errors available here: https://platform.openai.com/docs/guides/error-codes/api-errors
        except (
            openai.APIConnectionError,
            openai.APIError,
            openai.APIStatusError,
            openai.APIResponseValidationError,
            openai.AuthenticationError,
            openai.BadRequestError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.ConflictError,
            openai.UnprocessableEntityError,
        ) as e:
            # Fatal errors, do not retry
            msg = f"OpenAI API fatal error: {e}"
            logger.error(msg)
            return msg
        except (
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
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
                seed,
                current_attempt + 1,
            )
