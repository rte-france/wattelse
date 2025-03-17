#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import requests
from loguru import logger

from wattelse.api.security import FULL_ACCESS, RESTRICTED_ACCESS
from wattelse.api.security_client_utils import get_access_token

# Client credentials
CLIENT_ID = "admin"
CLIENT_SECRET = "57efb5729fed0f29185245e3cc282370397838ccebe29e4645e9fe9da1de0bab"
SCOPE = f"{FULL_ACCESS} {RESTRICTED_ACCESS}"

# API base URL
API_BASE_URL = "https://localhost:1234"


def call_protected_api():
    """Call a protected API endpoint using the access token"""
    token = get_access_token(
        api_base_url=API_BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scope=SCOPE,
    )

    # Use the token to call a protected endpoint
    headers = {"Authorization": f"Bearer {token}"}

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
