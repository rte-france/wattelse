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
    def get_articles(self, query: str, after: str, before: str) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines.

        Parameters
        ----------
        query: str
            keywords describing the request
        after: str
            date after which to consider articles, formatted as YYYY-MM-DD
        before: str
            date before which to consider articles, formatted as YYYY-MM-DD

        Returns
        -------
        A list of dict entries, each one describing an article
        """

        pass

    def get_articles_batch(self, queries_batch: List[List]) -> List[Dict]:
        """Requests the news data provider for a list of queries, collects a set of URLs to be parsed,
        return results as json lines"""
        articles = []
        for entry in queries_batch:
            logger.info(f"Processing query: {entry}")
            articles += self.get_articles(queries_batch[0], queries_batch[1], queries_batch[2])
        return articles

    def parse_article(self, url: str) -> Article:
        """Parses an article described by its URL"""
        article = self.article_parser.extract(url=url)
        return article

    @abstractmethod
    def _build_query(self, keywords: str, after: str, before: str) -> str:
        pass

    def store_articles(self, data: List[Dict], file_path: Path):
        """Store articles to a specific path as json lines"""
        if not data:
            logger.error("No data to be stored!")
            return -1
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