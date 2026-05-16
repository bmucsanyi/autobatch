from autobatch._config import FindConfig, validate_config
from autobatch._goals import Goal
from autobatch._probe import ProbeOutcome
from autobatch._selection import select_value


class TimingProbe:
    def __init__(self) -> None:
        self.calls = []

    def probe(self, value: int, *, timed: bool) -> ProbeOutcome:
        self.calls.append((value, timed))

        return ProbeOutcome(
            status="safe",
            value=value,
            steps_completed=1,
            timing_seconds=(1.0 / value,),
        )


class ExplodingCache:
    def __init__(self) -> None:
        self.calls = []

    def read_value(self) -> int | None:
        self.calls.append("read")
        msg = "timing goal must not read cache"
        raise AssertionError(msg)

    def write_value(self, value: int) -> None:
        _ = value
        self.calls.append("write")
        msg = "timing goal must not write cache"
        raise AssertionError(msg)


def noop_probe(value: int) -> None:
    _ = value


def config(goal: Goal) -> FindConfig:
    return validate_config(
        probe=noop_probe,
        values=(1, 2, 3),
        goal=goal,
        cache_key=("selection", goal.kind),
        warmup_steps=1,
        measure_steps=1,
        devices=[0],
    )


def test_timing_goal_does_not_touch_cache() -> None:
    probe = TimingProbe()

    value = select_value(
        config=config(Goal.best_value_rate()),
        probe=probe,
        cache=ExplodingCache(),
    )

    assert value == 3
    assert probe.calls == [(1, True), (2, True), (3, True)]
