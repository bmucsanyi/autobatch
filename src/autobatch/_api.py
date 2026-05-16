from collections.abc import Callable, Hashable, Iterable

from autobatch._cache import Cache
from autobatch._config import FindConfig, validate_config
from autobatch._distributed import (
    DistributedCache,
    TorchCollectives,
    distributed_is_initialized,
    validate_distributed_config,
    validate_distributed_selection,
)
from autobatch._errors import DistributedError
from autobatch._goals import Goal
from autobatch._probe import InProcessProbeRunner
from autobatch._search import _ProbeRunner
from autobatch._selection import select_value
from autobatch._staged import DistributedStagedProbeRunner
from autobatch._types import StagedProbe


def find(
    probe: Callable[[int], None] | StagedProbe,
    *,
    values: Iterable[int],
    goal: Goal,
    cache_key: Hashable,
    warmup_steps: int,
    measure_steps: int,
    devices: list[int],
) -> int:
    config = validate_config(
        probe=probe,
        values=tuple(values),
        goal=goal,
        cache_key=cache_key,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        devices=devices,
    )

    if distributed_is_initialized():
        return _find_distributed(config)

    return _find_single_process(config)


def _find_single_process(config: FindConfig) -> int:
    return select_value(
        config=config,
        probe=_runner_for_process(config),
        cache=_cache_for_process(config),
    )


def _find_distributed(config: FindConfig) -> int:
    collectives = TorchCollectives()

    try:
        validate_distributed_config(config, collectives)
        value = select_value(
            config=config,
            probe=_distributed_runner(config, collectives),
            cache=DistributedCache(Cache(config.cache_key), collectives),
        )

        if not config.goal.requires_timing():
            return value

        return validate_distributed_selection(value, collectives)
    finally:
        collectives.close()


def _distributed_runner(
    config: FindConfig,
    collectives: TorchCollectives,
) -> _ProbeRunner:
    if isinstance(config.probe, StagedProbe):
        return DistributedStagedProbeRunner(config, collectives)

    msg = "distributed find requires a StagedProbe"
    raise DistributedError(msg)


def _runner_for_process(config: FindConfig) -> _ProbeRunner:
    return InProcessProbeRunner(config)


def _cache_for_process(config: FindConfig) -> Cache:
    return Cache(config.cache_key)
