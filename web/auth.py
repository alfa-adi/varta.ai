"""
web/auth.py───────────
JWT authentication helpers for durable-session routes.
Anonymous live routes (POST /session/create, WS /ws/asr/*, /translate/*)
are NOT affected by this module.

Token format: HS256 compact JWT, base64url-encoded (no padding),
suitable as an RFC 6455 subprotocol token.

get_current_user          — raises 503 if AUTH_SECRET missing; 401 on bad token
get_current_user_optional — returns None on any failure (missing/invalid header)
authenticate_websocket    — reads Sec-WebSocket-Protocol: token-<jwt>
"""

import os
from typing import Optional

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

security = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    user_id: str
    roles: list[str] = []


def _get_secret() -> str:
    """
    Return AUTH_SECRET, raising 503 if not configured.
    Called at decode time (not at import time) so the server can start
    without AUTH_SECRET when durable routes are never hit.
    """
    secret = os.getenv("AUTH_SECRET")
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="AUTH_SECRET not configured; durable sessions unavailable",
        )
    return secret


def _decode(token: str) -> Optional[AuthenticatedUser]:
    """Decode and validate a HS256 JWT. Returns None on any failure."""
    secret = os.getenv("AUTH_SECRET")
    if not secret:
        return None   # called from optional paths; hard path goes through _get_secret
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return AuthenticatedUser(
            user_id=payload["sub"],
            roles=payload.get("roles", []),
        )
    except Exception:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> AuthenticatedUser:
    """
    Required auth dependency.
    Raises HTTP 503 if AUTH_SECRET is not configured.
    Raises HTTP 401 if header is missing or token is invalid.
    """
    secret = _get_secret()   # 503 if unconfigured
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    try:
        payload = jwt.decode(credentials.credentials, secret, algorithms=["HS256"])
        return AuthenticatedUser(
            user_id=payload["sub"],
            roles=payload.get("roles", []),
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> Optional[AuthenticatedUser]:
    """
    Optional auth dependency.
    Missing header → None.
    Invalid token  → None  (not 401; the route still proceeds anonymously).
    Durable identity for persist operations comes from the Redis session snapshot
    (durable_user_id), NOT from re-decoding the JWT on /translate/* requests.
    """
    if not credentials or not credentials.credentials:
        return None
    return _decode(credentials.credentials)


async def authenticate_websocket(
    raw_protocols: str,
) -> tuple[Optional[AuthenticatedUser], Optional[str]]:
    """
    Parse Sec-WebSocket-Protocol header for token-<jwt>.
    Returns (AuthenticatedUser, selected_protocol_string) on success,
    or (None, None) if no valid token found.

    The caller must pass the selected_protocol_string to
    websocket.accept(subprotocol=selected_protocol_string).
    """
    for proto in [p.strip() for p in raw_protocols.split(",") if p.strip()]:
        if proto.startswith("token-"):
            token = proto[6:]
            user = _decode(token)
            if user:
                return user, proto
    return None, None
