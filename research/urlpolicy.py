"""Admission policy for outbound fetches.

Keylane's own daemon listens on ``127.0.0.1:9100`` with no authentication, and
``GET /memories`` returns everything it has ever learned about the user. A model
that has just read a hostile page is one instruction away from being told to
fetch that URL, so every hop of every outbound fetch is checked here rather than
inside any one caller.

The rule is DSH's: resolve the host, and refuse if *any* answer is non-public.
Rejecting on any answer rather than on the address finally used means a name
that resolves to one public and one private address is refused outright.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_REDIRECTS = 5

_IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# The well-known NAT64 prefix. An address inside it carries a v4 destination in
# its low 32 bits, so a private v4 target can be smuggled through a v6 literal.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class UrlNotAllowed(ValueError):
    """A URL that may not be fetched. The message is shown to the model."""


def _embedded_v4(addr: _IpAddress) -> ipaddress.IPv4Address | None:
    """The IPv4 destination a v6 address stands for, if it stands for one."""
    if not isinstance(addr, ipaddress.IPv6Address):
        return None
    if addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    if addr.sixtofour is not None:
        return addr.sixtofour
    if addr in _NAT64:
        return ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    return None


def is_public_address(addr: _IpAddress) -> bool:
    """True only for an address that is routable on the public internet.

    Translation prefixes — IPv4-mapped, 6to4, NAT64 — are refused outright
    rather than unwrapped-and-admitted. A desktop assistant has no reason to
    reach anything through one, and every such form is a way to spell a private
    destination that a naive check reads as a public v6 address. The embedded
    target is still inspected first so the log names the real reason.
    """
    if _embedded_v4(addr) is not None:
        # Refused whichever target it carries. Python reports `::ffff:8.8.8.8`
        # as global (it delegates to the mapped v4) but `64:ff9b::808:808` as
        # not, so leaving this to `is_global` would admit one spelling of a
        # translated address and refuse another.
        return False
    if addr.is_loopback or addr.is_link_local or addr.is_multicast:
        return False
    if addr.is_private or addr.is_reserved or addr.is_unspecified:
        return False
    return bool(addr.is_global)


def _resolve(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise UrlNotAllowed(f"could not resolve host {host!r}: {exc}") from exc
    return sorted({str(info[4][0]) for info in infos})


def check_url(url: str) -> list[str]:
    """Admit one URL for fetching and return the addresses it resolved to.

    Raises :class:`UrlNotAllowed` with a message written for the model — it
    names the reason so the model can pick a different URL instead of retrying.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UrlNotAllowed(
            f"only http and https URLs can be fetched, not {scheme or 'a relative URL'!r}"
        )
    if parts.username or parts.password:
        raise UrlNotAllowed("URLs carrying credentials are not fetched")

    host = parts.hostname
    if not host:
        raise UrlNotAllowed("the URL has no host")
    port = parts.port or (443 if scheme == "https" else 80)

    # A bare IP literal skips DNS but still has to be public.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not is_public_address(literal):
            raise UrlNotAllowed(f"{host} is not a public address")
        return [str(literal)]

    addresses = _resolve(host, port)
    if not addresses:
        raise UrlNotAllowed(f"host {host!r} resolved to no addresses")
    for raw in addresses:
        if not is_public_address(ipaddress.ip_address(raw)):
            raise UrlNotAllowed(f"host {host!r} resolves to the non-public address {raw}")
    return addresses
