#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import base64
import functools
import re
import time
from urllib.parse import urlsplit

# Ref: https://stackoverflow.com/a/59023463/

_ENCODED_URL_PREFIX = "https://news.google.com/rss/articles/"
_ENCODED_URL_RE = re.compile(fr"^{re.escape(_ENCODED_URL_PREFIX)}(?P<encoded_url>[^?]+)")
_DECODED_URL_RE = re.compile(rb'^\x08\x13".+?(?P<primary_url>http[^\xd2]+)\xd2\x01')

@functools.lru_cache(2048)
def _decode_google_news_url(url: str) -> str:
    """Decode encoded Google News entry URLs."""
    match = _ENCODED_URL_RE.match(url)
    encoded_text = match.groupdict()["encoded_url"]  # type: ignore
    encoded_text += "==="  # Fix incorrect padding. Ref: https://stackoverflow.com/a/49459036/
    decoded_text = base64.urlsafe_b64decode(encoded_text)

    match = _DECODED_URL_RE.match(decoded_text)
    primary_url = match.groupdict()["primary_url"]  # type: ignore
    primary_url = primary_url.decode()
    return primary_url

def decode_google_news_url(url: str) -> str:  # Not cached because not all Google News URLs are encoded.
    """Return Google News entry URLs after decoding their encoding as applicable."""
    return _decode_google_news_url(url) if url.startswith(_ENCODED_URL_PREFIX) else url


def wait(secs):
    """wait decorator"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            time.sleep(secs)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def wait_if_seen_url(secs):
    """wait decorator based on URL cache: only waits max to secs for websites already seen"""

    def decorator(func):
        cache = {}

        def wrapper(*args, **kwargs):
            url = kwargs.get("url")
            if url is None:
                return func(*args, **kwargs)
            else:
                base_url = urlsplit(url).netloc
                last_call = cache.get(base_url)
                current_call = round(time.time() * 1000)
                if last_call is not None:
                    # sleep if recent call
                    delta = (current_call - last_call) / 1000
                    if delta < secs:
                        time.sleep(secs - delta)
                # update cache
                cache[base_url] = current_call
                return func(*args, **kwargs)

        return wrapper

    return decorator