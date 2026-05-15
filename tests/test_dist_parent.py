import pytest

from autobatch._dist_parent import aggregate_distributed_outcomes
from autobatch._errors import DistributedError
from autobatch._protocol import ProbeOutcome


def unsafe_outcome(value: int, reason: str) -> ProbeOutcome:
    return ProbeOutcome(status="unsafe", value=value, reason=reason)


def timeout_outcome(value: int, reason: str) -> ProbeOutcome:
    return ProbeOutcome(status="timeout", value=value, reason=reason)


def failed_outcome(value: int, reason: str, exception_message: str) -> ProbeOutcome:
    return ProbeOutcome(
        status="failed",
        value=value,
        reason=reason,
        exception_message=exception_message,
    )


def test_distributed_aggregate_marks_peer_timeout_after_oom_as_unsafe() -> None:
    outcomes = (
        unsafe_outcome(4, "cuda_oom"),
        timeout_outcome(4, "blocked"),
    )

    outcome = aggregate_distributed_outcomes(value=4, outcomes=outcomes)

    assert outcome.status == "unsafe"


def test_distributed_aggregate_returns_timeout_without_oom() -> None:
    outcomes = (
        timeout_outcome(4, "blocked"),
        timeout_outcome(4, "blocked"),
    )

    outcome = aggregate_distributed_outcomes(value=4, outcomes=outcomes)

    assert outcome.status == "timeout"


def test_distributed_aggregate_returns_non_oom_error() -> None:
    outcomes = (
        failed_outcome(4, "bug", "bad"),
        timeout_outcome(4, "blocked"),
    )

    outcome = aggregate_distributed_outcomes(value=4, outcomes=outcomes)

    assert outcome.status == "failed"
    assert outcome.exception_message == "bad"


def test_distributed_aggregate_rejects_empty_outcomes() -> None:
    with pytest.raises(DistributedError):
        aggregate_distributed_outcomes(value=4, outcomes=())
