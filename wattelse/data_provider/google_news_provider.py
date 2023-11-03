import urllib.parse
from typing import List, Dict, Optional

import dateparser
from joblib import Parallel, delayed
from loguru import logger
from pygooglenews import GoogleNews

from wattelse.data_provider.data_provider import DataProvider
from wattelse.data_provider.utils import wait, decode_google_news_url

PATTERN = "{QUERY}"
BEFORE = "+before:today"
AFTER = "+after:2000-01-01"
MAX_ARTICLES = 100


class GoogleNewsProvider(DataProvider):
    """News provider for Google News.
    Limitations:
        - since of results limited to 100
    """

    URL_ENDPOINT = f"https://news.google.com/rss/search?num={MAX_ARTICLES}&hl=fr&gl=FR&ceid=FR:fr&q={PATTERN}{BEFORE}{AFTER}"

    def __init__(self):
        super().__init__()
        self.gn = GoogleNews(lang = 'fr', country = 'FR')

    @wait(1)
    def get_articles(self, keywords: str, after: str, before: str, max_results: int) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        #FIXME: this may be blocked by google
        logger.info(f"Querying Google: {keywords}")
        result = self.gn.search(keywords, from_=after, to_=before)
        entries = result["entries"][:max_results]
        logger.info(f"Returned: {len(entries)} entries")

        # Number of parallel jobs you want to run (adjust as needed)
        num_jobs = -1 # all available cpus

        # Parallelize the loop using joblib
        results = Parallel(n_jobs=num_jobs)(
            delayed(self._parse_entry)(res) for res in entries
        )

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


    def _parse_entry(self, entry: Dict) -> Optional[Dict]:
        """Parses a Google news entry"""
        try:
            # NB. we do not use the title from Gnews as it is sometimes truncated
            link = entry["link"]
            url = decode_google_news_url(link)
            summary = entry["summary"]
            published = dateparser.parse(entry["published"]).strftime("%Y-%m-%d %H:%M:%S")
            text, title = self._get_text(url=url)
            text = self._filter_out_bad_text(text)
            if text is None or text=="":
                return None
            logger.debug(f"----- Title: {title},\tDate: {published}")
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
