"""Mailspring / MCP auth helpers."""

from __future__ import annotations

from app.plugins.mcp_client import normalize_auth_header


def test_normalize_auth_header_adds_bearer() -> None:
    assert normalize_auth_header("68a82d10-1934-4769-958c-5c07c60ae833") == (
        "Bearer 68a82d10-1934-4769-958c-5c07c60ae833"
    )


def test_normalize_auth_header_keeps_bearer() -> None:
    assert normalize_auth_header("Bearer abc") == "Bearer abc"


def test_normalize_auth_header_strips_authorization_prefix() -> None:
    assert normalize_auth_header("Authorization: Bearer abc") == "Bearer abc"
