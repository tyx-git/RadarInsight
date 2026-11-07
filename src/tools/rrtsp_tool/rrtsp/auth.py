# -*- coding: utf-8 -*-
"""R-RTSP L1 Digest 鉴权（RFC 7616 SHA-256 子集）。"""
from __future__ import annotations

import hashlib
import secrets
import time


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_nonce() -> str:
    return secrets.token_hex(16)


def parse_digest_header(value: str) -> dict[str, str]:
    """解析 Digest key=value, 支持双引号。"""
    body = value.strip()
    if body.lower().startswith("digest "):
        body = body[7:]
    out = {}
    key = ""
    token = []
    in_quotes = False
    i = 0
    while i < len(body):
        ch = body[i]
        if not key:
            if ch.isalnum() or ch == "_":
                start = i
                i += 1
                while i < len(body) and (body[i].isalnum() or body[i] == "_"):
                    i += 1
                key = body[start:i]
                continue
        elif ch == "=":
            i += 1
            if i < len(body) and body[i] == '"':
                in_quotes = True
                i += 1
                start = i
                while i < len(body):
                    if body[i] == '"':
                        break
                    i += 1
                out[key] = body[start:i]
                i += 1
            else:
                start = i
                while i < len(body) and body[i] != ",":
                    i += 1
                out[key] = body[start:i].strip()
            key = ""
        if ch == "," and not in_quotes:
            in_quotes = False
        i += 1
    return out


def digest_response(*, username: str, password: str, realm: str,
                    method: str, uri: str, nonce: str,
                    qop: str = "auth", nc: str = "00000001",
                    cnonce: str = "") -> str:
    ha1 = sha256(f"{username}:{realm}:{password}")
    ha2 = sha256(f"{method}:{uri}")
    return sha256(
        f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}")


def build_authorization(*, username: str, password: str, realm: str,
                        method: str, uri: str, nonce: str,
                        cnonce: str, nc: str = "00000001",
                        qop: str = "auth",
                        algorithm: str = "SHA-256") -> str:
    response = digest_response(
        username=username, password=password, realm=realm,
        method=method, uri=uri, nonce=nonce, qop=qop,
        nc=nc, cnonce=cnonce)
    return (
        'Digest '
        f'username="{username}", realm="{realm}", nonce="{nonce}", '
        f'uri="{uri}", response="{response}", qop={qop}, nc={nc}, '
        f'cnonce="{cnonce}", algorithm={algorithm}')


def validate_authorization(*, header: str, username: str, password: str,
                           realm: str, method: str, uri: str,
                           nonce: str, max_age_s: int = 86400,
                           issued_at: float | None = None) -> bool:
    creds = parse_digest_header(header)
    required = ("username", "realm", "nonce", "uri", "response",
                "qop", "nc", "cnonce", "algorithm")
    if any(key not in creds for key in required):
        return False
    if creds["username"] != username or creds["realm"] != realm:
        return False
    if creds["nonce"] != nonce or creds["uri"] != uri:
        return False
    if creds["qop"].lower() != "auth":
        return False
    if creds["algorithm"].upper().replace("-", "") != "SHA256":
        return False
    if issued_at is not None and time.time() - issued_at > max_age_s:
        return False

    expected = digest_response(
        username=username, password=password, realm=realm,
        method=method, uri=uri, nonce=nonce, qop=creds["qop"],
        nc=creds["nc"], cnonce=creds["cnonce"])
    return secrets.compare_digest(expected, creds["response"])
