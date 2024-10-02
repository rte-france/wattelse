#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from loguru import logger

from bertrend_apps.data_provider.data_provider import DataProvider
import feedparser


class CurebotProvider(DataProvider):
    """Class used to process exports from Curebot tool"""

    def __init__(self, curebot_export_file: Path = None, feed_url: str = None):
        super().__init__()
        self.data_file = curebot_export_file
        if self.data_file:
            self.df_dict = pd.read_excel(self.data_file, sheet_name=None, dtype=str)
        self.feed_url = feed_url

    def get_articles(
        self,
        query: str = None,
        after: str = None,
        before: str = None,
        max_results: int = None,
        language: str = "fr",
    ) -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        if self.feed_url:
            return self.parse_ATOM_feed()

        entries = []
        for k in self.df_dict.keys():
            entries += self.df_dict[k].to_dict(orient="records")
        results = [self._parse_entry(res) for res in entries]
        return [
            res for res in results if res is not None
        ]  # sanity check to remove errors

    def parse_ATOM_feed(self) -> List[Dict]:
        feed = feedparser.parse(self.feed_url)
        # Initialize an empty list to store the entries
        entries = []
        # Extract information for each entry
        for entry in feed.entries:
            link = entry.get("link", "")
            summary = entry.get("summary", "")
            published = entry.get("published", "")
            formatted_date = self.parse_date(published)
            text, title = self._get_text(url=link)
            if text is None or text == "":
                continue
            # Store the extracted information in a dictionary
            entry_info = {
                "title": title,
                "link": link,
                "url": link,
                "summary": summary,
                "text": text,
                "timestamp": formatted_date,
            }
            # Add the dictionary to the list
            entries.append(entry_info)

        return entries

    def _parse_entry(self, entry: Dict) -> Optional[Dict]:
        """Parses a Curebot news entry"""
        try:
            # NB. we do not use the title from Gnews as it is sometimes truncated
            link = entry["URL de la ressource"]
            url = link
            summary = entry["Contenu de la ressource"]
            published = self.parse_date(entry["Date de trouvaille"])
            text, title = self._get_text(url=url)
            if text is None or text == "":
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
            logger.error(
                str(e) + f"\nError occurred with text parsing of url in : {entry}"
            )
            return None
