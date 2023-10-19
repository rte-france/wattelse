from typing import List, Dict, Optional

import dateparser
from joblib import Parallel, delayed
from loguru import logger
from newscatcherapi import NewsCatcherApiClient

from wattelse.data_provider.data_provider import DataProvider
from wattelse.data_provider.utils import wait

API_KEY = "ajY3f53bIUECMKxE-Q5DvH2Etrq8QYYiC5GRWh8AIx4" # Free API key


class NewsCatcherProvider(DataProvider):
    """News provider for Bing News.
    Limitations:
        - depends on API KEY, with free version, request limited to 1/sec; content history limited to one month
    """

    def __init__(self):
        super().__init__()
        self.api_client = NewsCatcherApiClient(x_api_key=API_KEY)


    @wait(1)
    def get_articles(self, keywords: str, after: str, before: str, max_results: int) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""

        # Use the API to search articles
        logger.info(f"Querying NewsCatcher: {keywords}")
        result = self.api_client.get_search(q=keywords,
                                            lang='fr',
                                            page_size=max_results,
                                            from_=after,
                                            to_=before)

        entries = result["articles"][:max_results]
        logger.info(f"Returned: {len(entries)} entries")

        # Number of parallel jobs you want to run (adjust as needed)
        num_jobs = -1 # all available cpus

        # Parallelize the loop using joblib
        results = Parallel(n_jobs=num_jobs)(
            delayed(self._parse_entry)(res) for res in entries
        )
        return [res for res in results if res is not None]

    def _parse_entry(self, entry: Dict) -> Optional[Dict]:
        """Parses a NewsCatcher news entry"""
        try:
            title = entry["title"]
            link = entry["link"]
            url = link
            summary = entry["summary"]
            published = dateparser.parse(entry["published_date"]).strftime("%Y-%m-%d %H:%M:%S")
            text = self._get_text(url=url)
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