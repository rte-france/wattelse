from typing import List, Dict, Optional
from loguru import logger
import dateparser
import urllib.parse
import feedparser
from pygooglenews import GoogleNews
from newspaper import Article

from data_provider.data_provider import DataProvider
from data_provider.utils import wait, decode_google_news_url

PATTERN = "{QUERY}"
BEFORE = "+before:today"
AFTER = "+after:2000-01-01"
MAX_ARTICLES = 100


class GoogleNewsProvider(DataProvider):
    """News provider for Bing News.
    Limitations:
        - since of results limited to 12
        - hard to request specific dates
    """

    URL_ENDPOINT = f"https://news.google.com/rss/search?num={MAX_ARTICLES}&hl=fr&gl=FR&ceid=FR:fr&q={PATTERN}{BEFORE}{AFTER}"

    def __init__(self):
        super().__init__()
        self.gn = GoogleNews(lang = 'fr', country = 'FR')

    def get_articles_old(self, keywords: str, after: str, before: str) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        #FIXME: this may be blocked by google
        query = self._build_query(keywords, after, before)
        logger.debug(f"Querying Google: {query}")
        result = feedparser.parse(query)
        logger.debug(f"Returned: {len(result['entries'])} entries")

        results = [self._parse_entry(res) for res in result["entries"]]
        return [res for res in results if res is not None]

    def get_articles(self, keywords: str, after: str, before: str) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        #FIXME: this may be blocked by google
        logger.debug(f"Querying Google: {keywords}")
        result = self.gn.search(keywords, from_=after, to_=before)
        logger.debug(f"Returned: {len(result['entries'])} entries")

        results = [self._parse_entry(res) for res in result["entries"]]
        return [res for res in results if res is not None]

    def _build_query(self, keywords: str, after: str = None, before: str = None) -> str:
        query = self.URL_ENDPOINT.replace(PATTERN, f"{urllib.parse.quote(keywords)}")
        if after is None or after == "":
            query = query.replace(AFTER, "")
        else:
            query = query.replace(AFTER, f"+after:{after}")
        if before is None or before == "":
            query = query.replace(BEFORE, "")
        else:
            query = query.replace(BEFORE, f"+before:{before}")

        return query

    def _get_text(self, url: str) -> str:
        """Extracts text from an article URL"""
        logger.debug(f"Extracting text from {url}")
        try:
            article = self.parse_article(url)
            return article.cleaned_text
        except:
            # goose3 not working, trying with alternate parser
            logger.warning("Parsing of text failed with Goose3, trying newspaper3k")
            return self._get_text_alternate(url)

    def _get_text_alternate(self, url: str) -> str:
        """Extracts text from an article URL"""
        logger.debug(f"Extracting text from {url} with newspaper3k")
        article = Article(url)
        article.download()
        article.parse()
        return article.text

    def _filter_out_bad_text(self, text):
        if "[if" in text:
            logger.warning("Bad text: {text}")
            return None
        return text

    @wait(0.2)
    def _parse_entry(self, entry: Dict) -> Optional[Dict]:
        """Parses a Google news entry, uses wait decorator to force delay between 2 successive calls"""
        try:
            title = entry["title"]
            link = entry["link"]
            url = decode_google_news_url(link)
            print(url)
            summary = entry["summary"]
            published = dateparser.parse(entry["published"]).strftime("%Y-%m-%d %H:%M:%S")
            text = self._get_text(url)
            text = self._filter_out_bad_text(text)
            if text is None or text=="":
                return None
            return {
                "title": title,
                "summary": summary,
                "link": link,
                "url": url,
                "text": text,
                "timestamp": published,
            }
        except Exception as e:
            logger.error(str(e) + f"\nError occurred with text parsing of url in : {entry}")
            return None
