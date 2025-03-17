#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import json
import secrets

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import (
    OAuth2PasswordRequestForm,
    SecurityScopes,
)
from fastapi.security.oauth2 import OAuth2PasswordBearer
import jwt  # PyJWT
from datetime import datetime, timedelta, timezone
from typing import Annotated

from jwt import InvalidTokenError
from loguru import logger
from pydantic import BaseModel, ValidationError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def generate_hex_token(length: int = 32):
    """Generates a random hexadecimal string of the specified length."""
    return secrets.token_hex(length)


# Configuration (should be in environment variables)
# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = os.getenv("WATTELSE_SECRET_KEY", generate_hex_token())
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# Client registry - in production, store somewhere else - database, file...
CLIENT_REGISTRY_FILE = os.getenv(
    "CLIENT_REGISTRY_FILE", f"{os.path.expanduser('~')}/wattelse_client_registry.json"
)

DEFAULT_CLIENT_REGISTRY = {
    "admin": {
        "client_secret": generate_hex_token(),
        "scopes": ["admin", "full_access", "restricted"],
    },
    "wattelse": {
        "client_secret": generate_hex_token(),
        "scopes": ["full_access", "restricted"],
    },
    "opfab": {"client_secret": generate_hex_token(), "scopes": ["restricted"]},
}

RESTRICTED_ACCESS = "restricted_access"
FULL_ACCESS = "full_access"
ADMIN = "admin"

# Define available scopes with descriptions
SCOPES = {
    RESTRICTED_ACCESS: "Access limited to some endpoints",
    FULL_ACCESS: "Full access to endpoints except admin endpoints",
    ADMIN: "Admin access",
}

# Configure OAuth2
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes=SCOPES,
)


def load_client_registry():
    """Function to load client registry from JSON file"""
    try:
        with open(CLIENT_REGISTRY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create a default client registry file if it doesn't exist
        default_registry = DEFAULT_CLIENT_REGISTRY
        with open(CLIENT_REGISTRY_FILE, "w") as f:
            json.dump(default_registry, f, indent=4)
        return default_registry
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Client registry file is not valid JSON",
        )


def list_registered_clients():
    """View registered clients (without exposing secrets)"""
    client_registry = load_client_registry()
    # Remove sensitive information like client secrets
    clients_info = {}
    for client_id, client_data in client_registry.items():
        clients_info[client_id] = {"scopes": client_data["scopes"]}
    return clients_info


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    client_id: str | None = None
    scopes: list[str] = []


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """Creates access token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta or timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_client(
    security_scopes: SecurityScopes, token: Annotated[str, Depends(oauth2_scheme)]
):
    """Function to verify token and extract client data"""
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    try:
        payload = jwt.decode(token, key=SECRET_KEY, algorithms=[ALGORITHM])
        client_id = payload.get("sub")
        if client_id is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(client_id=client_id, scopes=token_scopes)
    except (InvalidTokenError, ValidationError):
        raise credentials_exception
    # Check if client still exists
    if client_id not in load_client_registry():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Client no longer exists",
            headers={"WWW-Authenticate": authenticate_value},
        )

    # Check if client has necessary scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            logger.error(f"{scope} not in {token_data.scopes}, not enough permissions")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required scope: {scope}",
                headers={"WWW-Authenticate": authenticate_value},
            )
    return token_data


def get_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """Implement the token endpoint for token generation"""
    client_id = (
        form_data.username
    )  # In client credentials, username field holds the client_id
    client_secret = form_data.password  # Password field holds the client_secret

    # Load client registry
    client_registry = load_client_registry()

    # Validate client
    if client_id not in client_registry:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    client = client_registry[client_id]
    if client["client_secret"] != client_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate scopes
    client_scopes = client["scopes"]

    requested_scopes = form_data.scopes

    # If no scopes requested, grant all available to client
    if not requested_scopes:
        scopes = client_scopes
    else:
        # Check if requested scopes are valid
        for scope in requested_scopes:
            if scope not in client_scopes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Client '{client_id}' does not have access to scope '{scope}'",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        scopes = requested_scopes

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": client_id, "scopes": scopes},
        expires_delta=access_token_expires,
    )

    logger.debug(f"Token created for client '{client_id}'")
    return Token(access_token=access_token, token_type="bearer")
