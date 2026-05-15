from collections.abc import Mapping

from autobatch._cache import Cache
from autobatch._config import FindConfig
from autobatch._search import _ProbeRunner, search


def select_value(
    *,
    config: FindConfig,
    probe: _ProbeRunner,
    cache: Cache,
    cache_identity: Mapping[str, object],
) -> int:
    cached = cache.read_value()

    if cached is not None and cached in config.domain.values:
        outcome = probe.probe(cached, timed=False)

        if outcome.status == "safe" and not config.goal.requires_timing():
            return cached

    value = search(
        domain=config.domain,
        goal=config.goal,
        runner=probe,
    )
    cache.write_value(value, identity=cache_identity)

    return value
