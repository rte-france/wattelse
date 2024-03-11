import configparser
from pathlib import Path
import requests
from loguru import logger


class OllamaAPI:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent / "ollama_api.cfg")
        self.port = config.get("API_CONFIG", "port")
        self.url = f"http://localhost:{self.port}/api/generate"
        self.model_name = config.get("API_CONFIG", "model_name")
        self.num_gpu = config.getint("API_CONFIG", "num_gpu")
        self.temperature = config.getfloat("API_CONFIG", "temperature")

    def get_api_model_name(self) -> str:
        """
        Return currently loaded model name in Ollama.
        """
        return self.model_name

    def generate(
            self,
            user_prompt: str,
            system_prompt: str = None,
            temperature: float = None,
            max_tokens: int = 512,
            stream: bool = False,
    ):
        """Uses the remote model (API) to generate the answer.

        Args:
            user_prompt (str): prompt to send to the model with role=user.
            system_prompt (str): prompt to send to the model with role=system. Useless if transform_prompt=False.
            temperature (float, optional): Temperature for generation.
            max_tokens (int, optional): Maximum tokens to be generated.
            stream (bool, optional): Whether to stream output. Defaults to False.

        Returns:
            str: Returns :
                    - full text once generated if stream=False
                    - Requests streaming response object if stream=True
        """
        logger.debug(f"Calling remote ollama service...")
        data = {
            "model": self.model_name,
            "prompt": user_prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature if temperature else self.temperature,
                "num_gpu": self.num_gpu,
            }
        }
        try:
            if stream:
                streaming_response = requests.post(self.url, json=data, stream=True)
                return streaming_response
            else:
                response = requests.post(self.url, json=data).json()
                return response["response"]
        except Exception as e:
            msg = f"Exception occurred with API call to {self.url}. Error: {e}"
            logger.error(msg)
            return msg
