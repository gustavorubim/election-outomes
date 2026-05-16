from __future__ import annotations

import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import FrameType
from typing import Any, TypeVar

T = TypeVar("T")


class BayesianTimeoutError(TimeoutError):
    """Raised when Bayesian inference exceeds the configured wall clock budget."""


@dataclass(frozen=True)
class FailoverPolicy:
    timeout_seconds: float | None = None
    fallback_order: tuple[str, ...] = (
        "previous_posterior_reuse",
        "bayes_svi_fallback",
        "kalman_legacy_fallback",
    )
    block_publication_on_fallback: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FailoverPolicy:
        bayesian = dict(config.get("bayesian", {}))
        nuts = dict(bayesian.get("nuts", {}))
        failover = dict(nuts.get("failover", config.get("failover", {})))
        raw_order = failover.get("fallback_order", cls.fallback_order)
        if isinstance(raw_order, str):
            order = tuple(part.strip() for part in raw_order.split(",") if part.strip())
        elif isinstance(raw_order, (list, tuple)):
            order = tuple(str(item) for item in raw_order if str(item).strip())
        else:
            order = cls.fallback_order
        timeout = nuts.get("wall_clock_timeout_seconds", failover.get("timeout_seconds"))
        return cls(
            timeout_seconds=float(timeout) if timeout is not None else None,
            fallback_order=order or cls.fallback_order,
            block_publication_on_fallback=bool(failover.get("block_publication_on_fallback", True)),
        )

    def with_timeout(self, timeout_seconds: float) -> FailoverPolicy:
        return replace(self, timeout_seconds=float(timeout_seconds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "fallback_order": list(self.fallback_order),
            "block_publication_on_fallback": self.block_publication_on_fallback,
        }


@dataclass(frozen=True)
class FailoverResult:
    result: Any
    audit: dict[str, Any]


def execute_with_failover(
    primary: Callable[[], T],
    fallback: Callable[[], T] | None,
    policy: FailoverPolicy,
    *,
    primary_engine: str,
) -> FailoverResult:
    """Run a primary inference callable under a wall-clock timeout.

    The function intentionally only catches timeout failures. Model exceptions still
    surface as implementation bugs unless the caller explicitly converts them into a
    fallback decision.
    """

    started = time.perf_counter()
    try:
        with _wall_clock_timeout(policy.timeout_seconds):
            result = primary()
    except BayesianTimeoutError as exc:
        elapsed = time.perf_counter() - started
        if fallback is None:
            raise
        fallback_label = policy.fallback_order[0] if policy.fallback_order else "fallback"
        fallback_result = fallback()
        return FailoverResult(
            result=fallback_result,
            audit={
                "status": "fallback_used",
                "primary_engine": primary_engine,
                "fallback_used": fallback_label,
                "reason": str(exc),
                "elapsed_seconds": round(float(elapsed), 6),
                "timeout_seconds": policy.timeout_seconds,
                "fallback_order": list(policy.fallback_order),
                "publication_blocked": policy.block_publication_on_fallback,
            },
        )
    elapsed = time.perf_counter() - started
    return FailoverResult(
        result=result,
        audit={
            "status": "completed",
            "primary_engine": primary_engine,
            "fallback_used": None,
            "reason": None,
            "elapsed_seconds": round(float(elapsed), 6),
            "timeout_seconds": policy.timeout_seconds,
            "fallback_order": list(policy.fallback_order),
            "publication_blocked": False,
        },
    )


def exercise_timeout_failover(policy: FailoverPolicy) -> dict[str, Any]:
    """Run a deterministic fixture audit that forces the first fallback path."""

    audit_policy = policy.with_timeout(0.01)

    def primary() -> str:
        time.sleep(0.05)
        return "primary_completed"

    def fallback() -> str:
        return "fallback_completed"

    result = execute_with_failover(
        primary,
        fallback,
        audit_policy,
        primary_engine="numpyro-nuts-fixture",
    )
    expected = policy.fallback_order[0] if policy.fallback_order else "fallback"
    return {
        "status": "exercised",
        "passed": result.audit.get("fallback_used") == expected,
        "audit_scope": "forced_timeout_fixture_not_forecast_fallback",
        "result": result.result,
        "policy": policy.to_dict(),
        "audit": result.audit,
    }


@contextmanager
def _wall_clock_timeout(timeout_seconds: float | None) -> Iterator[None]:
    if timeout_seconds is None or timeout_seconds <= 0:
        yield
        return
    if not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(_signum: int, _frame: FrameType | None) -> None:
        raise BayesianTimeoutError(
            f"Bayesian inference exceeded {float(timeout_seconds):.3f}s wall-clock timeout"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
