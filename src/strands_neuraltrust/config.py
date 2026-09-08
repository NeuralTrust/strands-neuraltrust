"""Explicit deployment configuration; no environment discovery or I/O."""

from __future__ import annotations

import ipaddress
import math
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .exceptions import TrustGuardConfigurationError


@dataclass(frozen=True, slots=True)
class TrustGuardConfig:
    """API-key configuration for one collector deployment.

    ``base_url`` is the deployment root, optionally including a path prefix; the
    client appends ``/v1/evaluate``. HTTP is accepted only for an explicitly opted
    in literal loopback address or ``localhost``. Environment proxies are ignored
    by owned transports. ``timeout`` bounds every HTTP phase and, for native async
    evaluation, the entire HTTP operation. Sync evaluation additionally checks
    elapsed time between response chunks, but an in-flight phase may finish later.
    """

    api_key: str = field(repr=False)
    base_url: str
    timeout: float = 5.0
    max_request_bytes: int = 1_048_576
    max_response_bytes: int = 1_048_576
    allow_local_http: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.api_key, str)
            or not self.api_key
            or len(self.api_key) > 4096
            or any(not 33 <= ord(char) <= 126 for char in self.api_key)
        ):
            raise TrustGuardConfigurationError("A valid collector API key is required.")
        if type(self.allow_local_http) is not bool:
            raise TrustGuardConfigurationError("The local HTTP option must be boolean.")
        if not _valid_timeout(self.timeout):
            raise TrustGuardConfigurationError("The evaluation timeout must be finite and positive.")
        for limit in (self.max_request_bytes, self.max_response_bytes):
            if type(limit) is not int or limit <= 0:
                raise TrustGuardConfigurationError("Evaluation byte limits must be positive integers.")
        object.__setattr__(self, "base_url", _deployment_url(self.base_url, self.allow_local_http))

    @property
    def evaluate_url(self) -> str:
        """The collector evaluation endpoint."""
        return self.base_url + "/v1/evaluate"


def _valid_timeout(value: float) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _deployment_url(value: str, allow_local_http: bool) -> str:
    message = "A valid HTTPS deployment base URL is required."
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(char) <= 32 or ord(char) == 127 for char in value)
    ):
        raise TrustGuardConfigurationError(message)
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        invalid = (
            not host
            or parsed.username is not None
            or parsed.password is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
            or "?" in value
            or "#" in value
            or (port is not None and not 1 <= port <= 65535)
            or parsed.scheme not in ("https", "http")
            or any(part in (".", "..") for part in parsed.path.split("/"))
            or "%" in parsed.netloc
            or "%" in parsed.path
        )
        if invalid:
            raise ValueError
        if parsed.scheme == "http":
            loopback = host == "localhost"
            if not loopback:
                try:
                    address = ipaddress.ip_address(host or "")
                    loopback = address.is_loopback and (
                        address.version == 4 or address == ipaddress.IPv6Address("::1")
                    )
                except ValueError:
                    loopback = False
            if not allow_local_http or not loopback:
                raise ValueError
        # IDNA conversion rejects malformed Unicode hostnames without DNS/network.
        (host or "").encode("idna")
        ascii_host = (host or "").encode("idna").decode("ascii")
        try:
            ipaddress.ip_address(ascii_host)
        except ValueError:
            if len(ascii_host) > 253 or not all(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                for label in ascii_host.rstrip(".").split(".")
            ):
                raise ValueError from None
        return value.rstrip("/")
    except (ValueError, UnicodeError):
        pass
    raise TrustGuardConfigurationError(message)
