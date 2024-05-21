#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
from pathlib import Path
from typing import List, Dict, Optional

import dateparser
import pandas as pd
from loguru import logger

from wattelse.data_provider.data_provider import DataProvider


class CurebotProvider(DataProvider):
    """Class used to process exports from Curebot tool"""

    def __init__(self, curebot_export_file: Path):
        super().__init__()
        self.data_file = curebot_export_file
        self.df_dict = pd.read_excel(self.data_file, sheet_name=None, dtype=str)

    def get_articles(self, query: str = None, after: str = None, before: str = None, max_results: int = None,
                     language: str = "fr") -> List[Dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        entries = []
        for k in self.df_dict.keys():
            entries += self.df_dict[k].to_dict(orient='records')
        return [self._parse_entry(res) for res in entries]

    def _parse_entry(self, entry: Dict) -> Optional[Dict]:
        """Parses a Curebot news entry"""
        try:
            # NB. we do not use the title from Gnews as it is sometimes truncated
            link = entry["URL de la ressource"]
            url = link
            summary = entry["Contenu de la ressource"]
            published = dateparser.parse(entry["Date de trouvaille"]).replace(microsecond=0).strftime(
                "%Y-%m-%d %H:%M:%S")
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
            logger.error(str(e) + f"\nError occurred with text parsing of url in : {entry}")
            return None