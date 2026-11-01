"""Safe, isolated redirect resolver.

This module resolves HTTP redirect chains while enforcing strict security
controls to prevent Server-Side Request Forgery (SSRF) and resource abuse.

Security model
--------------
1. **SSRF guard** — Every redirect target's hostname is resolved to A/AAAA
   records *before* following it.  Any non-global IP (loopback, link-local,
   RFC-1918, reserved) terminates the chain immediately with
   ``blocked_private_ip = True``.

2. **Hop cap** — The chain stops at ``max_hops`` (default 5, hard-capped at
   10).  Reaching the cap does NOT raise; ``hop_limit_reached`` is set.

3. **Subprocess isolation** — The actual network work runs in a separate
   ``multiprocessing.Process`` (not a thread).  The parent sets a wall-clock
   timeout and kills the worker if it exceeds it, preventing a misbehaving
   redirect target from blocking the API process indefinitely.

4. **Scheme guard** — Only ``http://`` and ``https://`` redirects are
   followed.  ``javascript:``, ``data:``, ``file:``, etc. terminate the chain
   with an error.

5. **No cookie jar, no JavaScript** — Uses ``requests`` with
   ``allow_redirects=False``; every hop is handled manually.

Usage
-----
    from src.redirect_resolver import resolve_redirects, RedirectChain
    chain: RedirectChain = resolve_redirects("https://bit.ly/example")
    print(chain.final_url, chain.hops)
"""

from __future__ import annotations

import ipaddress
import logging
import multiprocessing
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

from .config import REDIRECT_MAX_HOPS, REDIRECT_TIMEOUT

log = logging.getLogger("phishing_url_detector.redirect")

_HARD_HOP_CAP = 10
_ALLOWED_SCHEMES = {"http", "https"}

# User-agent string for redirect-following requests.
_USER_AGENT = "URLDomainPhishingDetector/1.0 (redirect resolver; defensive research)"


@dataclass
class RedirectHop:
    """One step in a redirect chain."""
    hop: int
    url: str
    status_code: int
    resolved_ip: str | None = None
    location: str | None = None


@dataclass
class RedirectChain:
    """Result of following a redirect chain from an initial URL."""
    initial_url: str
    hops: list[RedirectHop] = field(default_factory=list)
    final_url: str | None = None
    blocked_private_ip: bool = False
    hop_limit_reached: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_url": self.initial_url,
            "hops": [
                {
                    "hop": h.hop,
                    "url": h.url,
                    "status_code": h.status_code,
                    "resolved_ip": h.resolved_ip,
                    "location": h.location,
                }
                for h in self.hops
            ],
            "final_url": self.final_url,
            "blocked_private_ip": self.blocked_private_ip,
            "hop_limit_reached": self.hop_limit_reached,
            "error": self.error,
        }


def _is_private_ip(addr: str) -> bool:
    """Return True when the IP is not globally routable."""
    try:
        return not ipaddress.ip_address(addr.split("%")[0]).is_global
    except ValueError:
        return True  # unparseable → treat as private for safety


def _resolve_first_ip(hostname: str) -> str | None:
    """Return the first resolved IP for ``hostname``, or None on failure."""
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        if infos:
            return str(infos[0][4][0])
    except OSError:
        pass
    return None


def _ssrf_check(url: str) -> tuple[bool, str | None]:
    """Return (is_safe, resolved_ip).  is_safe=False blocks the request."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
    except Exception:
        return False, None
    if not hostname:
        return False, None
    # IP literals
    try:
        addr = ipaddress.ip_address(hostname.strip("[]"))
        return addr.is_global, str(addr)
    except ValueError:
        pass
    # Hostname — resolve and check each address
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError:
        return False, None
    if not infos:
        return False, None
    first_ip: str | None = None
    for info in infos:
        addr_str = str(info[4][0])
        if first_ip is None:
            first_ip = addr_str
        if _is_private_ip(addr_str):
            return False, addr_str  # any private addr blocks
    return True, first_ip


def _resolve_worker(initial_url: str, max_hops: int, timeout: float, result_queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """Worker function that runs in an isolated subprocess."""
    try:
        import requests as _requests  # noqa: PLC0415

        chain = RedirectChain(initial_url=initial_url)
        current_url = initial_url
        session = _requests.Session()
        session.headers.update({"User-Agent": _USER_AGENT})
        # Disable automatic redirect following — we handle each hop.
        session.max_redirects = 1

        for hop_index in range(1, max_hops + 1):
            # Validate scheme
            try:
                parsed = urlsplit(current_url)
                if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
                    chain.error = f"Redirect to disallowed scheme '{parsed.scheme}' at hop {hop_index}"
                    break
            except Exception as exc:
                chain.error = f"Could not parse URL at hop {hop_index}: {exc}"
                break

            # SSRF check
            is_safe, resolved_ip = _ssrf_check(current_url)
            if not is_safe:
                chain.blocked_private_ip = True
                chain.error = f"Redirect target resolves to a private/reserved IP at hop {hop_index}"
                break

            # Perform the request
            try:
                resp = session.get(current_url, allow_redirects=False, timeout=timeout, stream=True)
                # Do not download the body
                resp.close()
            except Exception as exc:
                chain.error = f"Request failed at hop {hop_index}: {type(exc).__name__}: {exc}"
                break

            location = resp.headers.get("Location", "")
            if location and not urlsplit(location).scheme:
                # Relative redirect
                location = urljoin(current_url, location)

            hop = RedirectHop(
                hop=hop_index,
                url=current_url,
                status_code=resp.status_code,
                resolved_ip=resolved_ip,
                location=location or None,
            )
            chain.hops.append(hop)

            if resp.status_code not in (301, 302, 303, 307, 308) or not location:
                chain.final_url = current_url
                break

            current_url = location

            if hop_index == max_hops:
                chain.hop_limit_reached = True
                chain.final_url = current_url

        result_queue.put(chain.to_dict())
    except Exception as exc:
        result_queue.put({"error": f"Worker crashed: {type(exc).__name__}: {exc}"})


def resolve_redirects(
    url: str,
    max_hops: int = REDIRECT_MAX_HOPS,
    timeout: float = REDIRECT_TIMEOUT,
) -> RedirectChain:
    """Follow ``url``'s redirect chain in an isolated subprocess.

    Returns a :class:`RedirectChain` regardless of outcome.  The ``error``
    field is set when the chain could not be completed cleanly.
    """
    max_hops = min(max_hops, _HARD_HOP_CAP)

    # Pre-flight scheme check (before spawning a process)
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            return RedirectChain(
                initial_url=url,
                error=f"Initial URL has disallowed scheme '{parsed.scheme}'",
            )
    except Exception as exc:
        return RedirectChain(initial_url=url, error=f"URL parse error: {exc}")

    # Pre-flight SSRF check on the initial URL
    is_safe, _ = _ssrf_check(url)
    if not is_safe:
        return RedirectChain(
            initial_url=url,
            blocked_private_ip=True,
            error="Initial URL resolves to a private/reserved IP",
        )

    # Spawn isolated worker
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()  # type: ignore[type-arg]
    worker = ctx.Process(
        target=_resolve_worker,
        args=(url, max_hops, timeout, result_queue),
        daemon=True,
    )
    worker.start()

    # Wait up to (max_hops * timeout + 2) seconds for the worker to finish
    budget = max_hops * timeout + 2.0
    worker.join(timeout=budget)

    if worker.is_alive():
        worker.kill()
        worker.join()
        return RedirectChain(
            initial_url=url,
            error=f"Redirect resolution timed out after {budget:.1f}s",
        )

    if result_queue.empty():
        return RedirectChain(
            initial_url=url,
            error="Worker finished without returning a result",
        )

    raw: dict[str, Any] = result_queue.get_nowait()
    if "error" in raw and len(raw) == 1:
        return RedirectChain(initial_url=url, error=raw["error"])

    return RedirectChain(
        initial_url=raw.get("initial_url", url),
        hops=[
            RedirectHop(
                hop=h["hop"],
                url=h["url"],
                status_code=h["status_code"],
                resolved_ip=h.get("resolved_ip"),
                location=h.get("location"),
            )
            for h in raw.get("hops", [])
        ],
        final_url=raw.get("final_url"),
        blocked_private_ip=raw.get("blocked_private_ip", False),
        hop_limit_reached=raw.get("hop_limit_reached", False),
        error=raw.get("error"),
    )
