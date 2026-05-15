import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from autobatch import _dist_parent
from autobatch._cache import Cache
from autobatch._config import FindConfig, validate_config
from autobatch._fingerprint import (
    distributed_fingerprint,
    environment_fingerprint,
    identity_key,
    runtime_fingerprint,
    workload_source_fingerprint,
)
from autobatch._goals import Goal
from autobatch._process import SubprocessProbeRunner
from autobatch._selection import select_value


def find(
    workload: str,
    *,
    values: Iterable[int],
    goal: Goal,
    reserve_fraction: float,
    reserve_bytes: int,
    warmup_steps: int,
    measure_steps: int,
    timeout_s: float,
    kwargs: Mapping[str, object],
    devices: object,
    cache_dir: str | Path,
) -> int:
    config = validate_config(
        workload=workload,
        values=tuple(values),
        goal=goal,
        kwargs=kwargs,
        reserve_fraction=reserve_fraction,
        reserve_bytes=reserve_bytes,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        timeout_s=timeout_s,
        devices=devices,
        cache_dir=cache_dir,
    )
    cache_identity = _identity_for_config(config)
    cache_key = identity_key(cache_identity)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return _dist_parent.find_distributed(
            config,
            cache_key=cache_key,
            cache_identity=cache_identity,
        )

    return select_value(
        config=config,
        probe=SubprocessProbeRunner(config),
        cache=Cache(config.cache_dir, cache_key),
        cache_identity=cache_identity,
    )


def cache_key_for_config(config: FindConfig) -> str:
    return identity_key(_identity_for_config(config))


def _identity_for_config(config: FindConfig) -> dict[str, object]:
    return {
        "workload": config.workload,
        "workload_source": workload_source_fingerprint(config.workload),
        "kwargs": config.kwargs,
        "domain": {"values": list(config.domain.values)},
        "goal": config.goal.to_json(),
        "memory": {
            "reserve_fraction": config.reserve_fraction,
            "reserve_bytes": config.reserve_bytes,
        },
        "steps": {
            "warmup_steps": config.warmup_steps,
            "measure_steps": config.measure_steps,
        },
        "timeout_s": config.timeout_s,
        "devices": config.devices,
        "runtime": runtime_fingerprint(),
        "environment": environment_fingerprint(),
        "distributed": distributed_fingerprint(),
    }
