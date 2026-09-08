"""Content-safe failures raised at the TrustGuard protection boundary."""


class TrustGuardError(Exception):
    """Base class for a terminal TrustGuard protection failure.

    Applications should supply only fixed, content-free messages when constructing
    these errors. The integration never includes evaluator bodies or credentials.
    """

    def __init__(self, message: str = "TrustGuard protection failed.") -> None:
        super().__init__(message)


class TrustGuardBlocked(TrustGuardError):
    """The evaluator denied protected content or an action."""


class TrustGuardApprovalRequired(TrustGuardError):
    """The evaluator requested approval, which this integration cannot provide."""


class TrustGuardProtocolError(TrustGuardError):
    """The evaluator did not return a valid, supported decision."""


class TrustGuardUnavailable(TrustGuardError):
    """Evaluation could not complete within its transport/resource bounds."""


class TrustGuardAuthenticationError(TrustGuardError):
    """The configured collector credential was not accepted."""


class TrustGuardConfigurationError(TrustGuardError, ValueError):
    """The local configuration or evaluation request is invalid."""


class TrustGuardTransformError(TrustGuardError):
    """A transformation could not be applied safely."""


class TrustGuardUnsupportedContentError(TrustGuardError):
    """Content falls outside the integration's qualified content subset."""


class TrustGuardStateError(TrustGuardError):
    """A closed resource or unsafe conversation cannot be continued."""
