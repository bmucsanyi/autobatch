import pytest

from autobatch._config import FindConfig, validate_config
from autobatch._domain import Domain
from autobatch._errors import (
    InvalidConfigurationError,
    NoSafeValueError,
    ProbeTimeoutError,
    WorkloadError,
)
from autobatch._goals import Goal
from autobatch._protocol import ProbeOutcome
from autobatch._search import search


class FakeProbe:
    def __init__(
        self,
        safe_values: set[int],
        timings: dict[int, tuple[float, ...]],
    ) -> None:
        self.safe_values = safe_values
        self.timings = timings
        self.calls = []

    def probe(self, value: int, *, timed: bool) -> ProbeOutcome:
        self.calls.append(value)

        if value not in self.safe_values:
            return ProbeOutcome(status="unsafe", value=value, reason="test")

        timing_seconds = self.timings[value] if timed else ()

        return ProbeOutcome(
            status="safe",
            value=value,
            steps_completed=1,
            timing_seconds=timing_seconds,
        )


def make_config(domain: Domain, goal: Goal) -> FindConfig:
    return validate_config(
        workload="tests.fixtures.probes:step",
        values=domain.values,
        goal=goal,
        kwargs={},
        reserve_fraction=0.05,
        reserve_bytes=0,
        warmup_steps=1,
        measure_steps=1,
        timeout_s=1.0,
        devices=[0],
        cache_dir="cache",
    )


def run_search(settings: FindConfig, probe: FakeProbe) -> int:
    return search(
        settings.domain,
        settings.goal,
        runner=probe,
    )


def test_largest_safe_binary_search() -> None:
    settings = make_config(Domain(range(1, 9)), Goal.largest_safe())
    probe = FakeProbe({1, 2, 3, 4, 5}, {})

    value = run_search(settings, probe)

    assert value == 5
    assert len(probe.calls) < 8


def test_smallest_safe_binary_search() -> None:
    settings = make_config(Domain(range(1, 9)), Goal.smallest_safe())
    probe = FakeProbe({5, 6, 7, 8}, {})

    value = run_search(settings, probe)

    assert value == 5
    assert len(probe.calls) < 8


def test_largest_safe_raises_when_none_safe() -> None:
    settings = make_config(Domain(range(1, 5)), Goal.largest_safe())

    with pytest.raises(NoSafeValueError):
        run_search(settings, FakeProbe(set(), {}))


def test_fastest_step_checks_every_candidate() -> None:
    settings = make_config(
        Domain([16, 32, 64]),
        Goal.fastest_step(tie="smaller"),
    )
    probe = FakeProbe({16, 32, 64}, timings={16: (0.4,), 32: (0.2,), 64: (0.3,)})

    value = run_search(settings, probe)

    assert value == 32
    assert probe.calls == [16, 32, 64]


def test_best_value_rate_uses_value_over_seconds() -> None:
    settings = make_config(
        Domain([1, 2, 3]),
        Goal.best_value_rate(tie="larger"),
    )
    probe = FakeProbe({1, 2, 3}, timings={1: (1.0,), 2: (0.75,), 3: (2.0,)})

    value = run_search(settings, probe)

    assert value == 2


def test_target_rate_checks_declared_direction() -> None:
    goal = Goal.smallest_value_meeting_rate(
        target_per_second=1.0, direction="increasing"
    )
    settings = make_config(Domain([1, 2]), goal)
    probe = FakeProbe({1, 2}, timings={1: (1.0,), 2: (4.0,)})

    with pytest.raises(InvalidConfigurationError):
        run_search(settings, probe)


def test_failed_probe_raises_workload_error() -> None:
    settings = make_config(Domain([1]), Goal.largest_safe())

    class FailedProbe:
        @staticmethod
        def probe(value: int, *, timed: bool) -> ProbeOutcome:
            _ = timed

            return ProbeOutcome(
                status="failed",
                value=value,
                reason="bug",
                exception_message="boom",
            )

    with pytest.raises(WorkloadError):
        search(
            settings.domain,
            settings.goal,
            runner=FailedProbe(),
        )


def test_timeout_probe_raises_probe_timeout() -> None:
    settings = make_config(Domain([1]), Goal.largest_safe())

    class TimeoutProbe:
        @staticmethod
        def probe(value: int, *, timed: bool) -> ProbeOutcome:
            _ = timed

            return ProbeOutcome(status="timeout", value=value, reason="timeout")

    with pytest.raises(ProbeTimeoutError):
        search(
            settings.domain,
            settings.goal,
            runner=TimeoutProbe(),
        )
