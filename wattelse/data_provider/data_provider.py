import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple

import jsonlines
import pandas as pd
from goose3 import Goose
from loguru import logger
from newspaper import Article

from wattelse.data_provider.utils import wait_if_seen_url

# Ensures to write with +rw for both user and groups
os.umask(0o002)

# List of URLs we do not want to have results from (ex. obsolete or not pertinent)
BLACKLISTED_URL = [
    "www.filiere-3e.fr",
    "www.courrier-picard.fr",
    "www.aisnenouvelle.fr",
    "matinlibre.com",
    "www.lest-eclair.fr",
    "www.lunion.fr",
    "www.nordlittoral.fr",
    # journaux sur abonnements
    "www.ouest-france.fr",
    # download impossible
]


class DataProvider(ABC):
    def __init__(self):
        self.article_parser = Goose()
        # set 'standard' user agent
        self.article_parser.config.browser_user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2)"
        )

    @abstractmethod
    def get_articles(
        self, query: str, after: str, before: str, max_results: int
    ) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines.

        Parameters
        ----------
        query: str
            keywords describing the request
        after: str
            date after which to consider articles, formatted as YYYY-MM-DD
        before: str
            date before which to consider articles, formatted as YYYY-MM-DD
        max_results: int
            Maximum number of results per request

        Returns
        -------
        A list of dict entries, each one describing an article
        """

        pass

    def get_articles_batch(
        self, queries_batch: List[List], max_results: int
    ) -> List[Dict]:
        """Requests the news data provider for a list of queries, collects a set of URLs to be parsed,
        return results as json lines"""
        articles = []
        for entry in queries_batch:
            logger.info(f"Processing query: {entry}")
            articles += self.get_articles(entry[0], entry[1], entry[2], max_results)
        # remove duplicates
        articles = [dict(t) for t in {tuple(d.items()) for d in articles}]
        logger.info(f"Collected {len(articles)} articles")
        return articles

    def parse_article(self, url: str) -> Article:
        """Parses an article described by its URL"""
        article = self.article_parser.extract(url=url)
        return article

    def store_articles(self, data: List[Dict], file_path: Path):
        """Store articles to a specific path as json lines"""
        if not data:
            logger.error("No data to be stored!")
            return -1
        with jsonlines.open(file_path, "w") as writer:
            writer.write_all(data)

        logger.info(f"Data stored to {file_path} [{len(data)} entries].")

    def load_articles(self, file_path: Path) -> pd.DataFrame:
        """Read articles serialized as json files and provide an associated dataframe"""
        with open(file_path, "r") as f:
            with jsonlines.Reader(f) as reader:
                data = reader.read()
                return pd.DataFrame(data)

    @wait_if_seen_url(0.2)
    def _get_text(self, url: str) -> Tuple[str, str]:
        """Extracts text and (clean) title from an article URL"""
        if any(ele in url for ele in BLACKLISTED_URL):
            logger.warning(f"Source of {url} is blacklisted!")
            return None

        logger.debug(f"Extracting text from {url}")
        try:
            article = self.parse_article(url)
            return article.cleaned_text, article.title
        except:
            # goose3 not working, trying with alternate parser
            logger.warning("Parsing of text failed with Goose3, trying newspaper3k")
            return self._get_text_alternate(url)

    def _get_text_alternate(self, url: str) -> Tuple[str, str]:
        """Extracts text from an article URL"""
        logger.debug(f"Extracting text from {url} with newspaper3k")
        article = Article(url)
        article.download()
        article.parse()
        return article.text, article.title

    def _filter_out_bad_text(self, text):
        if (
            "[if" in text
            or "javascript" in text
            or "cookie" in text
            or "pour lire la suite, rejoignez notre communauté d'abonnés" in text
        ):
            logger.warning(f"Bad text: {text}")
            return None
        return text
