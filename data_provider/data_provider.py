from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict
from goose3 import Goose, Article
import jsonlines
from loguru import logger


class DataProvider(ABC):

    def __init__(self):
        self.article_parser = Goose()

    @abstractmethod
    def get_articles(self, keywords: List[str]) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        pass

    def parse_article(self, url: str) -> Article:
        """Parses an article described by its URL"""
        article = self.article_parser.extract(url=url)
        return article

    def store_articles(self, data: List[Dict], file_path: Path):
        """Store articles to a specific path as json lines"""
        with open(file_path, "a+") as f:
            with jsonlines.Writer(f) as writer:
                writer.write(data)
        logger.debug(f"Data stored to {file_path}.")