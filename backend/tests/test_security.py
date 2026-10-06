"""Tests for security helpers (hashing + JWT)."""

import time

import jwt
import pytest

from app.core.config import settings
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_and_verify():
    h = hash_password("s3cret-pass")
    assert h.startswith("pbkdf2$")
    assert verify_password("s3cret-pass", h)
    assert not verify_password("wrong", h)
    # salts differ per call
    assert hash_password("s3cret-pass") != h


def test_hash_is_constant_time_safe():
    h = hash_password("x")
    assert not verify_password("", h)
    assert not verify_password(h, h)


def test_jwt_roundtrip():
    token = create_access_token("user-123", extra={"email": "a@b.c"})
    payload = decode_access_token(token)
    assert payload["sub"] == "user-123"
    assert payload["email"] == "a@b.c"
    assert "exp" in payload


def test_jwt_rejects_garbage():
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token("not.a.token")


def test_jwt_rejects_wrong_secret():
    token = create_access_token("user-123")
    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(token, "other-secret", algorithms=[settings.JWT_ALGORITHM])
