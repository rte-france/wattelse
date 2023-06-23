from abc import ABC, abstractmethod
from typing import List
from goose3 import Goose, Article


class DataProvider(ABC):

    def __init__(self):
        self.article_parser = Goose()

    @abstractmethod
    def get_articles(self, keywords: List[str]):
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        pass

    def parse_article(self, url: str) -> Article:
        """Parses an article described by its URL"""
        article = self.article_parser.extract(url=url)
        return article