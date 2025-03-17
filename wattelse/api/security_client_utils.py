#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import requests


def get_access_token(
    api_base_url: str, client_id: str, client_secret: str, scope: str = None
) -> str:
    """Obtain an access token using the client credentials flow"""
    token_url = f"{api_base_url}/token"
    response = requests.post(
        token_url,
        data={
            "username": client_id,  # Send client_id as username
            "password": client_secret,  # Send client_secret as password
            "scope": scope,  # Requesting specific scopes
        },
        verify=False,
    )

    # NB in requests, verify=False to avoid problems of certificate not generated by a trusted CA

    if response.status_code == 200:
        token_data = response.json()
        return token_data["access_token"]
    else:
        raise Exception(
            f"Failed to obtain token: {response.status_code} {response.text}"
        )
