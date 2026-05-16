import pytest

import autobatch
import autobatch._api as api
from autobatch._cache import Cache
from autobatch._config import FindConfig, validate_config
from autobatch._errors import InvalidConfigurationError
from autobatch._goals import Goal
from autobatch._probe import ProbeOutcome


class FakeRunner:
    def __init__(self, safe_until: int) -> None:
        self.safe_until = safe_until
        self.calls = []

    def probe(self, value: int, *, timed: bool) -> ProbeOutcome:
        _ = timed
        self.calls.append(value)

        if value <= self.safe_until:
            return ProbeOutcome(status="safe", value=value, steps_completed=1)

        return ProbeOutcome(status="unsafe", value=value, reason="cuda_oom")


class TimingRunner:
    def __init__(self) -> None:
        self.calls = []

    def probe(self, value: int, *, timed: bool) -> ProbeOutcome:
        self.calls.append((value, timed))

        return ProbeOutcome(
            status="safe",
            value=value,
            timing_seconds=(1.0 / value,),
            steps_completed=1,
        )


def noop_probe(value: int) -> None:
    _ = value


def _make_config(values: tuple[int, ...], goal: Goal, cache_key: object) -> FindConfig:
    return validate_config(
        probe=noop_probe,
        values=values,
        goal=goal,
        cache_key=cache_key,
        warmup_steps=1,
        measure_steps=1,
        devices=[0],
    )


def test_find_returns_integer_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner(3)
    monkeypatch.setattr(api, "_runner_for_process", lambda _: runner)

    value = autobatch.find(
        noop_probe,
        values=[1, 2, 3, 4],
        goal=autobatch.Goal.largest_safe(),
        cache_key=("api-boundary", "a"),
        warmup_steps=1,
        measure_steps=1,
        devices=[0],
    )

    assert value == 3
    assert value in runner.calls


def test_find_searches_again_when_cached_value_revalidates_as_unsafe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner(safe_until=2)
    monkeypatch.setattr(api, "_runner_for_process", lambda _: runner)

    cache_key = ("api-revalidate", "a")
    config = _make_config((1, 2, 3, 4), Goal.largest_safe(), cache_key)
    Cache(config.cache_key).write_value(3)

    value = autobatch.find(
        noop_probe,
        values=[1, 2, 3, 4],
        goal=autobatch.Goal.largest_safe(),
        cache_key=cache_key,
        warmup_steps=1,
        measure_steps=1,
        devices=[0],
    )

    assert value == 2
    assert runner.calls[0] == 3


def test_find_rejects_cached_value_outside_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner(safe_until=4)
    monkeypatch.setattr(api, "_runner_for_process", lambda _: runner)

    cache_key = ("api-cache-domain", "a")
    config = _make_config((1, 2, 3), Goal.largest_safe(), cache_key)
    Cache(config.cache_key).write_value(4)

    with pytest.raises(InvalidConfigurationError, match="outside"):
        autobatch.find(
            noop_probe,
            values=[1, 2, 3],
            goal=autobatch.Goal.largest_safe(),
            cache_key=cache_key,
            warmup_steps=1,
            measure_steps=1,
            devices=[0],
        )

    assert runner.calls == []


def test_timing_goal_ignores_cached_value_and_scans_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = TimingRunner()
    monkeypatch.setattr(api, "_runner_for_process", lambda _: runner)

    goal = Goal.best_value_rate()
    cache_key = ("api-timing-cache", "a")
    config = _make_config((1, 2, 3), goal, cache_key)
    Cache(config.cache_key).write_value(1)

    value = autobatch.find(
        noop_probe,
        values=[1, 2, 3],
        goal=goal,
        cache_key=cache_key,
        warmup_steps=1,
        measure_steps=1,
        devices=[0],
    )

    assert value == 3
    assert runner.calls == [(1, True), (2, True), (3, True)]
