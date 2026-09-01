"""Fetch and extract readable text from web pages."""

from __future__ import annotations

import html
import logging
import re
from urllib.parse import urlparse

import httpx

from daemon.config import research_settings
from research.urlpolicy import MAX_REDIRECTS, UrlNotAllowed, check_url

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Keylane/1.0"
)

_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANKLINES = re.compile(r"\n{3,}")

_JS_HOSTS = ("reddit.com", "x.com", "twitter.com", "medium.com")


def strip_html(markup: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", markup)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", text, flags=re.IGNORECASE)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANKLINES.sub("\n\n", text).strip()


def extract_readable(markup: str, *, url: str = "") -> tuple[str, str]:
    title = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", markup, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = strip_html(title_match.group(1))

    try:
        import trafilatura
        from trafilatura.settings import use_config

        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
        extracted = trafilatura.extract(
            markup,
            url=url or None,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            config=config,
        )
        meta = trafilatura.extract_metadata(markup, default_url=url or None)
        if meta and getattr(meta, "title", None):
            title = str(meta.title).strip() or title
        if extracted and extracted.strip():
            return title or url, extracted.strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("trafilatura failed: %s", exc)

    return title or url, strip_html(markup)


def _text_is_thin(text: str, *, min_chars: int = 200) -> bool:
    body = (text or "").strip()
    if len(body) < min_chars:
        return True
    lower = body.lower()
    markers = (
        "enable javascript",
        "access denied",
        "403 forbidden",
        "just a moment",
        "sign in to continue",
    )
    return any(m in lower for m in markers)


def _host_prefers_js(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host.endswith(h) or host == h for h in _JS_HOSTS)


async def _read_capped(resp: httpx.Response, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` so an enormous body cannot exhaust memory."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.aiter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break
    return b"".join(chunks)[:max_bytes]


async def _get_following_redirects(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int,
) -> tuple[str, bytes, int]:
    """GET ``url``, revalidating the policy at every redirect hop.

    httpx's own ``follow_redirects`` would send us to whatever ``Location``
    says, which is exactly the hole ``check_url`` exists to close — a public
    host is free to redirect to ``127.0.0.1``. So redirects are walked here and
    each hop is admitted on its own.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_url(current)
        resp = await client.send(
            client.build_request("GET", current),
            stream=True,
            follow_redirects=False,
        )
        try:
            if resp.is_redirect:
                location = resp.headers.get("location", "")
                if not location:
                    raise UrlNotAllowed("redirect without a Location header")
                current = str(resp.next_request.url) if resp.next_request else location
                continue
            body = await _read_capped(resp, max_bytes)
            return str(resp.url), body, resp.status_code
        finally:
            await resp.aclose()
    raise UrlNotAllowed(f"more than {MAX_REDIRECTS} redirects starting at {url}")


async def fetch_page(url: str) -> dict[str, str]:
    cfg = research_settings()
    timeout = float(cfg.get("timeout_seconds", 25))
    max_chars = int(cfg.get("max_page_chars", 12000))
    max_bytes = int(cfg.get("max_page_bytes", 4_000_000))

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        final_url, body, status = await _get_following_redirects(
            client, url, max_bytes=max_bytes
        )

    markup = body.decode("utf-8", errors="replace")
    title, text = extract_readable(markup, url=final_url)

    if _text_is_thin(text) and (
        _host_prefers_js(final_url) or cfg.get("playwright_enabled")
    ):
        pw = await _playwright_fetch(final_url, cfg)
        if pw:
            title, text = pw.get("title", title), pw.get("text", text)

    truncated = len(text) > max_chars
    if truncated:
        # Say what to do about it, the way DSH does: a bare ellipsis tells the
        # model nothing, and it will usually just refetch the same URL.
        text = (
            text[:max_chars]
            + "\n\n(Content truncated. Fetch a more specific URL or section "
            "for the full text.)"
        )

    return {
        "url": final_url,
        "title": title,
        "text": text,
        "status_code": str(status),
        "truncated": "true" if truncated else "false",
    }


async def _playwright_fetch(url: str, cfg: dict) -> dict[str, str] | None:
    pw_url = str(cfg.get("playwright_url", ""))
    if not pw_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(pw_url, json={"url": url})
            resp.raise_for_status()
            data = resp.json()
            return {"title": data.get("title", ""), "text": data.get("text", "")}
    except Exception as exc:  # noqa: BLE001
        logger.debug("playwright fetch failed: %s", exc)
        return None
