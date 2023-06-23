import base64
from typing import List, Dict, Optional
from loguru import logger
import dateparser
import urllib.parse
import feedparser

from data_provider.data_provider import DataProvider
from data_provider.utils import wait

PATTERN = "{QUERY}"
MAX_ARTICLES = 100

class GoogleNewsProvider(DataProvider):
    """News provider for Bing News.
    Limitations:
        - since of results limited to 12
        - hard to request specific dates
    """
    URL_ENDPOINT = f"https://news.google.com/rss/search?num={MAX_ARTICLES}&hl=fr&gl=FR&ceid=FR:fr&q={PATTERN}"

    def __init__(self):
        super().__init__()

    def get_articles(self, keywords: List[str]) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        query = self._build_query(keywords)
        logger.debug(f"Querying Google: {query}")
        result = feedparser.parse(query)
        logger.debug(f"Returned: {len(result['entries'])} entries")

        results = [self._parse_entry(res) for res in result["entries"]]
        return [res for res in results if res is not None]

    def _build_query(self, keywords: List) -> str:
        kw = urllib.parse.quote(" ".join(keywords))
        return self.URL_ENDPOINT.replace(PATTERN, f"{kw}")

    def _clean_url(self, url) -> str:
        """Clean encoded URLs returned by Bing news such as "http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;aid=&amp;tid=649475a6257945d6900378c8310bcfde&amp;url=https%3a%2f%2fwww.lemondeinformatique.fr%2factualites%2flire-avec-schema-gpt-translator-datastax-automatise-la-creation-de-pipelines-de-donnees-90737.html&amp;c=15009376565431680830&amp;mkt=fr-fr"
        """
        try:
            base64_url = url.replace("https://news.google.com/rss/articles/", "").split("?")[0]
            # fix padding
            base64_url += "=" * ((4 - len(base64_url) % 4) % 4)
            #actual_url = base64.b64decode(base64_url)[5:-3].decode('utf-8')
            actual_url = base64.b64decode(base64_url)[4:-3].decode("ISO-8859-1")
            if actual_url.startswith("\x01"): # workaround
                actual_url = actual_url.split("\x01")[1]
            return actual_url
        except IndexError:
            # fallback (the URL does not match the expected pattern)
            return url

    def _get_text(self, url: str) -> str:
        """Extracts text from an article URL"""
        logger.debug(f"Extracting text from {url}")
        article = self.parse_article(url)
        return article.cleaned_text

    @wait(1)
    def _parse_entry(self, entry: Dict) -> Optional[Dict]:
        """Parses a Bing news entry, uses wait decorator to force delay between 2 successive calls"""
        try:
            title = entry["title"]
            link = entry["link"]
            url = self._clean_url(link)
            print(url)
            summary = entry["summary"]
            published = dateparser.parse(entry["published"]).strftime("%Y-%m-%d %H:%M:%S")
            text = self._get_text(url)
            return {'title': title, 'summary': summary, 'link': link, 'url': url, 'text': text,
                    'timestamp': published}
        except Exception as e:
            logger.error(str(e) + f"\nError occurred with text parsing of url in : {entry}")
            return None
