#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Security
from fastapi.security import (
    OAuth2PasswordRequestForm,
)
from loguru import logger

from wattelse.api.security import (
    Token,
    get_current_client,
    get_token,
    TokenData,
    list_registered_clients,
    RESTRICTED_ACCESS,
    FULL_ACCESS,
    ADMIN,
    is_authorized_for_group,
)

app = FastAPI()


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    return get_token(form_data)


@app.get("/list_registered_clients")
async def list_clients():
    return list_registered_clients()


# Protected endpoints with different scopes
@app.get("/api/data", summary="Read data (requires restricted or full_access scope)")
async def read_data(
    current_client: TokenData = Security(get_current_client, scopes=[RESTRICTED_ACCESS])
):
    return {
        "data": "This is protected data",
        "client": current_client.client_id,
        "access_level": "read",
    }


@app.post("/api/data", summary="Write data (requires full_access scope)")
async def write_data(
    current_client: TokenData = Security(get_current_client, scopes=[FULL_ACCESS])
):
    logger.info(f"Group check: {is_authorized_for_group(current_client, 'wattelse')}")
    return {
        "status": "Data successfully written",
        "client": current_client.client_id,
        "access_level": "write",
    }


@app.get("/api/admin", summary="Admin endpoint (requires admin scope)")
async def admin_endpoint(
    current_client: TokenData = Security(get_current_client, scopes=[ADMIN])
):
    return {
        "status": "Access to admin functionality",
        "client": current_client.client_id,
        "access_level": "admin",
    }


# Public endpoint (no auth)
@app.get("/api/public")
async def public_endpoint():
    return {"message": "This is a public endpoint"}


# Start the FastAPI application
if __name__ == "__main__":
    uvicorn.run(
        "wattelse.api.secure_api_example.secure_api:app",
        port=1234,
        host="localhost",
        reload=True,
        ssl_keyfile="../key.pem",
        ssl_certfile="../cert.pem",
    )
