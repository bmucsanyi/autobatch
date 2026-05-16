import math
from typing import Protocol

from autobatch._domain import Domain
from autobatch._errors import (
    InvalidConfigurationError,
    NoSafeValueError,
    ProbeError,
)
from autobatch._goals import Goal
from autobatch._probe import ProbeOutcome
from autobatch._timing import median_seconds


class _ProbeRunner(Protocol):
    def probe(self, value: int, *, timed: bool) -> ProbeOutcome: ...


def search(
    domain: Domain,
    goal: Goal,
    *,
    runner: _ProbeRunner,
) -> int:
    if goal.kind == "largest_safe":
        return _largest_safe(domain=domain, probe=runner)

    if goal.kind == "smallest_safe":
        return _smallest_safe(domain=domain, probe=runner)

    if goal.kind == "fastest_step":
        return _fastest_step(domain=domain, probe=runner)

    if goal.kind == "best_value_rate":
        return _best_value_rate(domain=domain, probe=runner)

    if goal.kind == "smallest_value_meeting_rate":
        return _target_rate(domain=domain, goal=goal, probe=runner)

    if goal.kind == "largest_value_under_latency":
        return _target_latency(domain=domain, goal=goal, probe=runner)

    msg = f"unsupported goal: {goal.kind}"
    raise InvalidConfigurationError(msg)


def _largest_safe(*, domain: Domain, probe: _ProbeRunner) -> int:
    low = 0
    high = len(domain) - 1
    best = None

    while low <= high:
        midpoint = (low + high) // 2
        value = domain[midpoint]
        outcome = probe.probe(value, timed=False)

        if _outcome_is_safe(outcome):
            best = value
            low = midpoint + 1
        else:
            high = midpoint - 1

    if best is None:
        msg = "no safe value exists in the declared domain"
        raise NoSafeValueError(msg)

    return best


def _smallest_safe(*, domain: Domain, probe: _ProbeRunner) -> int:
    low = 0
    high = len(domain) - 1
    best = None

    while low <= high:
        midpoint = (low + high) // 2
        value = domain[midpoint]
        outcome = probe.probe(value, timed=False)

        if _outcome_is_safe(outcome):
            best = value
            high = midpoint - 1
        else:
            low = midpoint + 1

    if best is None:
        msg = "no safe value exists in the declared domain"
        raise NoSafeValueError(msg)

    return best


def _fastest_step(
    *,
    domain: Domain,
    probe: _ProbeRunner,
) -> int:
    best_value = None
    best_seconds = None

    for value in domain.values:
        outcome = probe.probe(value, timed=True)

        if outcome.status == "unsafe":
            continue

        raise_for_bad_outcome(outcome)
        seconds = _timing_value(outcome)
        better = best_seconds is None or seconds < best_seconds

        if better:
            best_value = value
            best_seconds = seconds

    if best_value is None:
        msg = "no safe value exists in the declared domain"
        raise NoSafeValueError(msg)

    return best_value


def _best_value_rate(
    *,
    domain: Domain,
    probe: _ProbeRunner,
) -> int:
    best_value = None
    best_rate = None

    for value in domain.values:
        outcome = probe.probe(value, timed=True)

        if outcome.status == "unsafe":
            continue

        raise_for_bad_outcome(outcome)
        rate = value / _timing_value(outcome)
        better = best_rate is None or rate > best_rate

        if better:
            best_value = value
            best_rate = rate

    if best_value is None:
        msg = "no safe value exists in the declared domain"
        raise NoSafeValueError(msg)

    return best_value


def _target_rate(*, domain: Domain, goal: Goal, probe: _ProbeRunner) -> int:
    if goal.target_per_second is None:
        msg = "target_per_second is required"
        raise InvalidConfigurationError(msg)

    observations = []

    for value in domain.values:
        outcome = probe.probe(value, timed=True)

        if outcome.status == "unsafe":
            observations.append((value, None))
            continue

        raise_for_bad_outcome(outcome)
        observations.append((value, value / _timing_value(outcome)))

    for value, rate in observations:
        if rate is not None and rate >= goal.target_per_second:
            return value

    msg = "no safe value meets the target rate"
    raise NoSafeValueError(msg)


def _target_latency(*, domain: Domain, goal: Goal, probe: _ProbeRunner) -> int:
    if goal.max_seconds is None:
        msg = "max_seconds is required"
        raise InvalidConfigurationError(msg)

    observations = []

    for value in domain.values:
        outcome = probe.probe(value, timed=True)

        if outcome.status == "unsafe":
            observations.append((value, None))
            continue

        raise_for_bad_outcome(outcome)
        observations.append((value, _timing_value(outcome)))

    best = None

    for value, seconds in observations:
        if seconds is not None and seconds <= goal.max_seconds:
            best = value

    if best is None:
        msg = "no safe value meets the latency target"
        raise NoSafeValueError(msg)

    return best


def _outcome_is_safe(outcome: ProbeOutcome) -> bool:
    if outcome.status == "safe":
        return True

    if outcome.status == "unsafe":
        return False

    raise_for_bad_outcome(outcome)

    return False


def raise_for_bad_outcome(outcome: ProbeOutcome) -> None:
    if outcome.status == "failed":
        detail = outcome.exception_message

        if detail is None:
            detail = _failure_reason(outcome)

        raise ProbeError(detail)

    if outcome.status not in {"safe", "unsafe"}:
        msg = f"unknown probe outcome status: {outcome.status}"
        raise ProbeError(msg)


def _failure_reason(outcome: ProbeOutcome) -> str:
    if outcome.reason is None:
        msg = "failed outcome is missing a reason"
        raise ProbeError(msg)

    return outcome.reason


def _timing_value(outcome: ProbeOutcome) -> float:
    seconds = median_seconds(outcome.timing_seconds)

    if not math.isfinite(seconds) or seconds <= 0:
        msg = "timing sample must be finite and positive"
        raise ProbeError(msg)

    return seconds
