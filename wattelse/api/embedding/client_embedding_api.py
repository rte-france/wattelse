import configparser
from pathlib import Path
import requests
import json
from loguru import logger
from typing import List

import numpy as np
from numpy import ndarray


class EmbeddingAPI:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent / "embedding_api.cfg")
        self.port = config.get("EMBEDDING_API_CONFIG", "port")
        self.url = f'http://localhost:{self.port}'
        self.model_name = config.get("EMBEDDING_API_CONFIG", "model_name")

    def get_api_model_name(self) -> str:
        """
        Return currently loaded model name in Embedding API.
        """
        return self.model_name
    
    def encode(self, text: str | List[str], show_progress_bar: bool = False) -> ndarray:
        if type(text)==str:
            text = [text]
        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")
        logger.debug(f"Computing embeddings...")
        response = requests.post(self.url+'/encode', data=json.dumps({'text': text, 'show_progress_bar': show_progress_bar}))
        if response.status_code == 200:
            embeddings = np.array(response.json()["embeddings"])
            logger.debug(f"Computing embeddings done")
            return embeddings
        else:
            logger.error(f"Error: {response.status_code}")
        

