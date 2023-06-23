from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

import pandas as pd
from goose3 import Goose, Article
import jsonlines
from loguru import logger


class DataProvider(ABC):

    def __init__(self):
        self.article_parser = Goose()

    @abstractmethod
    def get_articles(self, query: str) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        pass

    def get_articles_batch(self, queries_batch: List[str]) -> List[Dict]:
        """Requests the news data provider for a list of queries, collects a set of URLs to be parsed, return results as json lines"""
        articles = []
        for query in queries_batch:
            logger.info(f"Processing query: {query}")
            articles += self.get_articles(query)
        return articles

    def parse_article(self, url: str) -> Article:
        """Parses an article described by its URL"""
        article = self.article_parser.extract(url=url)
        return article

    @abstractmethod
    def _build_query(self, keywords: str) -> str:
        pass

    def store_articles(self, data: List[Dict], file_path: Path):
        """Store articles to a specific path as json lines"""
        with open(file_path, "a+") as f:
            with jsonlines.Writer(f) as writer:
                writer.write(data)
        logger.debug(f"Data stored to {file_path}.")

    def load_articles(self, file_path: Path) -> pd.DataFrame:
        """Read articles serialized as json files and provide an associated dataframe"""
        with open(file_path, "r") as f:
            with jsonlines.Reader(f) as reader:
                data = reader.read()
                return pd.DataFrame(data)