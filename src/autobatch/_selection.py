from typing import Protocol

from autobatch._config import FindConfig
from autobatch._errors import InvalidConfigurationError
from autobatch._search import _ProbeRunner, raise_for_bad_outcome, search


class _Cache(Protocol):
    def read_value(self) -> int | None: ...

    def write_value(self, value: int) -> None: ...


def select_value(
    *,
    config: FindConfig,
    probe: _ProbeRunner,
    cache: _Cache,
) -> int:
    cached = cache.read_value()

    if not config.goal.requires_timing() and cached is not None:
        if cached not in config.domain.values:
            msg = "cached value is outside the declared domain"
            raise InvalidConfigurationError(msg)

        outcome = probe.probe(cached, timed=config.goal.requires_timing())

        if outcome.status == "safe":
            return cached

        raise_for_bad_outcome(outcome)

    value = search(
        domain=config.domain,
        goal=config.goal,
        runner=probe,
    )
    cache.write_value(value)

    return value
