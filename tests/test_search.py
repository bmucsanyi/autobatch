import pytest

from autobatch._config import FindConfig, validate_config
from autobatch._domain import Domain
from autobatch._errors import (
    NoSafeValueError,
    ProbeError,
)
from autobatch._goals import Goal
from autobatch._probe import ProbeOutcome
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


def noop_probe(value: int) -> None:
    _ = value


def make_config(domain: Domain, goal: Goal) -> FindConfig:
    return validate_config(
        probe=noop_probe,
        values=domain.values,
        goal=goal,
        cache_key=("search-config", domain.values, goal.kind),
        warmup_steps=1,
        measure_steps=1,
        devices=[0],
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
        Goal.fastest_step(),
    )
    probe = FakeProbe({16, 32, 64}, timings={16: (0.4,), 32: (0.2,), 64: (0.3,)})

    value = run_search(settings, probe)

    assert value == 32
    assert probe.calls == [16, 32, 64]


def test_best_value_rate_uses_value_over_seconds() -> None:
    settings = make_config(
        Domain([1, 2, 3]),
        Goal.best_value_rate(),
    )
    probe = FakeProbe({1, 2, 3}, timings={1: (1.0,), 2: (0.75,), 3: (2.0,)})

    value = run_search(settings, probe)

    assert value == 2


def test_fastest_step_keeps_first_value_on_equal_timing() -> None:
    settings = make_config(
        Domain([16, 32]),
        Goal.fastest_step(),
    )
    probe = FakeProbe({16, 32}, timings={16: (0.2,), 32: (0.2,)})

    value = run_search(settings, probe)

    assert value == 16


def test_best_value_rate_keeps_first_value_on_equal_rate() -> None:
    settings = make_config(
        Domain([1, 2]),
        Goal.best_value_rate(),
    )
    probe = FakeProbe({1, 2}, timings={1: (1.0,), 2: (2.0,)})

    value = run_search(settings, probe)

    assert value == 1


def test_target_rate_returns_smallest_safe_value_meeting_target() -> None:
    goal = Goal.smallest_value_meeting_rate(target_per_second=2.0)
    settings = make_config(Domain([1, 2, 3]), goal)
    probe = FakeProbe(
        {1, 2, 3},
        timings={1: (1.0,), 2: (0.75,), 3: (1.0,)},
    )

    value = run_search(settings, probe)

    assert value == 2


def test_target_latency_returns_largest_safe_value_under_limit() -> None:
    goal = Goal.largest_value_under_latency(max_seconds=0.8)
    settings = make_config(Domain([1, 2, 3]), goal)
    probe = FakeProbe(
        {1, 2, 3},
        timings={1: (0.2,), 2: (0.5,), 3: (0.9,)},
    )

    value = run_search(settings, probe)

    assert value == 2


def test_failed_probe_raises_probe_error() -> None:
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

    with pytest.raises(ProbeError):
        search(
            settings.domain,
            settings.goal,
            runner=FailedProbe(),
        )


def test_timing_goal_rejects_nan_timing_sample() -> None:
    settings = make_config(Domain([1]), Goal.fastest_step())
    probe = FakeProbe({1}, timings={1: (float("nan"),)})

    with pytest.raises(ProbeError, match="finite"):
        run_search(settings, probe)


def test_timing_goal_rejects_infinite_timing_sample() -> None:
    settings = make_config(Domain([1]), Goal.best_value_rate())
    probe = FakeProbe({1}, timings={1: (float("inf"),)})

    with pytest.raises(ProbeError, match="finite"):
        run_search(settings, probe)
