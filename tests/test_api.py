from pathlib import Path

import pytest

import autobatch
import autobatch._api as api
from autobatch._cache import Cache
from autobatch._config import FindConfig, validate_config
from autobatch._goals import Goal
from autobatch._protocol import ProbeOutcome


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


def _drop_torchrun_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def _make_config(values: tuple[int, ...], goal: Goal, cache_dir: Path) -> FindConfig:
    return validate_config(
        workload="tests.fixtures.probes:step",
        values=values,
        goal=goal,
        kwargs={},
        reserve_fraction=0.05,
        reserve_bytes=0,
        warmup_steps=1,
        measure_steps=1,
        timeout_s=1.0,
        devices=[0],
        cache_dir=cache_dir,
    )


def test_find_returns_integer_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = FakeRunner(3)
    monkeypatch.setattr(api, "SubprocessProbeRunner", lambda _: runner)
    _drop_torchrun_env(monkeypatch)

    value = autobatch.find(
        workload="tests.fixtures.probes:step",
        values=[1, 2, 3, 4],
        goal=autobatch.Goal.largest_safe(),
        kwargs={},
        reserve_fraction=0.05,
        reserve_bytes=0,
        warmup_steps=1,
        measure_steps=1,
        timeout_s=1.0,
        devices=[0],
        cache_dir=tmp_path,
    )

    assert value == 3
    assert value in runner.calls


def test_find_searches_again_when_cached_value_revalidates_as_unsafe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = FakeRunner(safe_until=2)
    monkeypatch.setattr(api, "SubprocessProbeRunner", lambda _: runner)
    _drop_torchrun_env(monkeypatch)

    config = _make_config((1, 2, 3, 4), Goal.largest_safe(), tmp_path)
    cache_key = api.cache_key_for_config(config)
    Cache(tmp_path, cache_key).write_value(3, identity={})

    value = autobatch.find(
        workload="tests.fixtures.probes:step",
        values=[1, 2, 3, 4],
        goal=autobatch.Goal.largest_safe(),
        kwargs={},
        reserve_fraction=0.05,
        reserve_bytes=0,
        warmup_steps=1,
        measure_steps=1,
        timeout_s=1.0,
        devices=[0],
        cache_dir=tmp_path,
    )

    assert value == 2
    assert runner.calls[0] == 3


def test_timing_goal_ignores_cache_for_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = TimingRunner()
    monkeypatch.setattr(api, "SubprocessProbeRunner", lambda _: runner)
    _drop_torchrun_env(monkeypatch)

    goal = Goal.best_value_rate(tie="larger")
    config = _make_config((1, 2, 3), goal, tmp_path)
    cache_key = api.cache_key_for_config(config)
    Cache(tmp_path, cache_key).write_value(1, identity={})

    value = autobatch.find(
        workload="tests.fixtures.probes:step",
        values=[1, 2, 3],
        goal=goal,
        kwargs={},
        reserve_fraction=0.05,
        reserve_bytes=0,
        warmup_steps=1,
        measure_steps=1,
        timeout_s=1.0,
        devices=[0],
        cache_dir=tmp_path,
    )

    assert value == 3
    assert [call[0] for call in runner.calls] == [1, 1, 2, 3]
