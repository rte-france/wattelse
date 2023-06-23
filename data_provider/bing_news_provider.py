from typing import List, Dict, Optional
from loguru import logger
import dateparser
import urllib.parse
import feedparser

from data_provider.data_provider import DataProvider
from data_provider.utils import wait

PATTERN = "{QUERY}"


class BingNewsProvider(DataProvider):
    URL_ENDPOINT = f"https://www.bing.com/news/search?q={PATTERN}&format=rss"

    def __init__(self):
        super().__init__()

    def get_articles(self, keywords: List[str]) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        query = self._build_query(keywords)
        logger.debug(f"Querying Bing: {query}")
        result = feedparser.parse(query)
        logger.debug(f"Returned: {len(result['entries'])} entries")

        results = [self._parse_entry(res) for res in result["entries"]]
        return [res for res in results if res is not None]

    def _build_query(self, keywords: List) -> str:
        kw = urllib.parse.quote(" ".join(keywords))
        return self.URL_ENDPOINT.replace(PATTERN, f"{kw}")

    def _clean_url(self, bing_url) -> str:
        """Clean encoded URLs returned by Bing news such as "http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;aid=&amp;tid=649475a6257945d6900378c8310bcfde&amp;url=https%3a%2f%2fwww.lemondeinformatique.fr%2factualites%2flire-avec-schema-gpt-translator-datastax-automatise-la-creation-de-pipelines-de-donnees-90737.html&amp;c=15009376565431680830&amp;mkt=fr-fr"
        """
        try:
            clean_url = bing_url.split("url=")[1].split("&  ")[0]
            return urllib.parse.unquote(clean_url)
        except IndexError:
            # fallback (the URL does not match the expected pattern)
            return bing_url

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
            summary = entry["summary"]
            published = dateparser.parse(entry["published"]).strftime("%Y-%m-%d %H:%M:%S")
            text = self._get_text(link)
            return {'title': title, 'summary': summary, 'link': link, 'url': url, 'text': text,
                    'timestamp': published}
        except Exception as e:
            logger.error(str(e) + f"\nError occurred with text parsing of: {url}")
            return None
