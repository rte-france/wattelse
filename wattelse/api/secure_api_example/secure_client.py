#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import os

import requests
from loguru import logger

from wattelse.api.security import FULL_ACCESS, RESTRICTED_ACCESS
from wattelse.api.security_client_utils import get_access_token

# Client credentials
CLIENT_ID = "wattelse"
CLIENT_SECRET = os.getenv("WATTELSE_CLIENT_SECRET", None)
SCOPE = None  # use default for clients

# API base URL
API_BASE_URL = "https://localhost:1234"


def _get_headers(url, client_id, client_secret):
    """Helper function to get headers with authentification token"""
    token = get_access_token(
        api_base_url=url,
        client_id=client_id,
        client_secret=client_secret,
    )
    # Use the token to call a protected endpoint
    headers = {"Authorization": f"Bearer {token}"}
    return headers


def call_protected_api():
    """Call a protected API endpoint using the access token"""
    headers = _get_headers(API_BASE_URL, CLIENT_ID, CLIENT_SECRET)

    # Read data (requires read:data scope)
    read_response = requests.get(
        f"{API_BASE_URL}/api/data", headers=headers, verify=False
    )
    logger.info(f"Read data response: {read_response.json()}")

    # Write data (requires write:data scope)
    write_response = requests.post(
        f"{API_BASE_URL}/api/data", headers=headers, json={"some": "data"}, verify=False
    )
    logger.info(f"Write data response: {write_response.json()}")


if __name__ == "__main__":
    call_protected_api()
