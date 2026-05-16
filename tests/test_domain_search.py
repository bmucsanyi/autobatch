import pytest

from autobatch._config import FindConfig, validate_config
from autobatch._domain import Domain
from autobatch._errors import NoSafeValueError
from autobatch._goals import Goal
from autobatch._probe import ProbeOutcome
from autobatch._search import search


class Probe:
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def probe(self, value: int, *, timed: bool) -> ProbeOutcome:
        _ = timed

        if value <= self.threshold:
            return ProbeOutcome(status="safe", value=value, steps_completed=1)

        return ProbeOutcome(status="unsafe", value=value, reason="test")


def noop_probe(value: int) -> None:
    _ = value


def config(domain: Domain) -> FindConfig:
    return validate_config(
        probe=noop_probe,
        values=domain.values,
        goal=Goal.largest_safe(),
        cache_key=("domain-search", domain.values),
        warmup_steps=1,
        measure_steps=1,
        devices=[0],
    )


def test_largest_safe_binary_search_returns_boundary() -> None:
    settings = config(Domain(range(1, 8)))
    value = search(
        settings.domain,
        settings.goal,
        runner=Probe(5),
    )

    assert value == 5


def test_largest_safe_raises_when_minimum_is_unsafe() -> None:
    settings = config(Domain(range(1, 4)))

    with pytest.raises(NoSafeValueError):
        search(
            settings.domain,
            settings.goal,
            runner=Probe(0),
        )
