"""Optional R-RTSP Digest SHA-256 authentication."""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Mapping


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_digest(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    body = value.strip()
    if body.lower().startswith("digest "):
        body = body[7:]
    result: dict[str, str] = {}
    index = 0
    while index < len(body):
        while index < len(body) and body[index] in " ,\t":
            index += 1
        start = index
        while index < len(body) and (body[index].isalnum() or body[index] == "_"):
            index += 1
        key = body[start:index].lower()
        if not key or index >= len(body) or body[index] != "=":
            break
        index += 1
        if index < len(body) and body[index] == '"':
            index += 1
            start = index
            while index < len(body) and body[index] != '"':
                index += 1
            result[key] = body[start:index]
            index += int(index < len(body))
        else:
            start = index
            while index < len(body) and body[index] != ",":
                index += 1
            result[key] = body[start:index].strip()
    return result


class DigestAuthenticator:
    """Validate the SHA-256 digest subset defined by R-RTSP v1."""

    def __init__(
        self,
        *,
        mode: str = "open",
        username: str = "viewer",
        password: str = "",
        realm: str = "AeroSight",
        nonce_ttl_seconds: int = 86400,
    ) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"open", "digest"}:
            raise ValueError("authentication mode must be 'open' or 'digest'")
        if normalized == "digest" and not username:
            raise ValueError("digest authentication requires a username")
        self.mode = normalized
        self.username = str(username)
        self.password = str(password)
        self.realm = str(realm)
        self.nonce_ttl_seconds = int(nonce_ttl_seconds)
        self.nonce = secrets.token_hex(16)
        self.issued_at = time.time()

    def challenge(self) -> str:
        return (
            f'Digest realm="{self.realm}", nonce="{self.nonce}", '
            'algorithm=SHA-256, qop="auth"'
        )

    def valid(self, headers: Mapping[str, str], method: str, uri: str) -> bool:
        if self.mode == "open":
            return True
        values = _parse_digest(headers.get("authorization"))
        required = {
            "username", "realm", "nonce", "uri", "response",
            "qop", "nc", "cnonce", "algorithm",
        }
        if not required.issubset(values):
            return False
        if values["username"] != self.username or values["realm"] != self.realm:
            return False
        if values["nonce"] != self.nonce or values["uri"] != uri:
            return False
        if values["qop"].lower() != "auth":
            return False
        if values["algorithm"].upper().replace("-", "") != "SHA256":
            return False
        if time.time() - self.issued_at > self.nonce_ttl_seconds:
            return False
        ha1 = _sha256(f"{self.username}:{self.realm}:{self.password}")
        ha2 = _sha256(f"{method}:{uri}")
        expected = _sha256(
            f"{ha1}:{values['nonce']}:{values['nc']}:{values['cnonce']}"
            f":{values['qop']}:{ha2}"
        )
        return secrets.compare_digest(expected, values["response"])


__all__ = ["DigestAuthenticator"]
