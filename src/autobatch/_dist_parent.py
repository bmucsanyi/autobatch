import dataclasses
import os
import socket
from collections.abc import Mapping

import torch.distributed as dist

from autobatch._cache import Cache
from autobatch._config import FindConfig
from autobatch._errors import DistributedError
from autobatch._process import make_worker_request, run_worker_process
from autobatch._protocol import ProbeOutcome
from autobatch._selection import select_value


def find_distributed(
    config: FindConfig,
    *,
    cache_key: str,
    cache_identity: Mapping[str, object],
) -> int:
    _ensure_parent_group()

    value = select_value(
        config=config,
        probe=_DistributedProbe(config),
        cache=Cache(config.cache_dir, cache_key),
        cache_identity=cache_identity,
    )
    values = _empty_object_list(dist.get_world_size())
    dist.all_gather_object(values, value)

    selected_values = tuple(_selected_value_from_gathered(item) for item in values)

    if len(set(selected_values)) != 1:
        msg = "parent ranks selected different values"
        raise DistributedError(msg)

    return value


class _DistributedProbe:
    def __init__(self, config: FindConfig) -> None:
        self.config = config
        self.candidate_id = 0

    def probe(self, value: int, *, timed: bool) -> ProbeOutcome:
        self.candidate_id += 1
        port = _broadcast_candidate_port()
        request = make_worker_request(self.config, value, timed=timed)
        request.master_port = port
        request.candidate_id = self.candidate_id
        env = _candidate_env(port=port, candidate_id=self.candidate_id)
        local = run_worker_process(request, self.config.timeout_s, env=env)
        gathered = _empty_object_list(dist.get_world_size())
        dist.all_gather_object(gathered, local.to_json())
        outcomes = tuple(_outcome_from_gathered(item) for item in gathered)

        return aggregate_distributed_outcomes(value=value, outcomes=outcomes)


def aggregate_distributed_outcomes(
    *,
    value: int,
    outcomes: tuple[ProbeOutcome, ...] | list[ProbeOutcome],
) -> ProbeOutcome:
    if len(outcomes) == 0:
        msg = "no distributed outcomes were reported"
        raise DistributedError(msg)

    unsafe = next((o for o in outcomes if o.status == "unsafe"), None)

    if unsafe is not None:
        return dataclasses.replace(unsafe, value=value)

    failed = next((o for o in outcomes if o.status == "failed"), None)

    if failed is not None:
        return dataclasses.replace(failed, value=value)

    timeout = next((o for o in outcomes if o.status == "timeout"), None)

    if timeout is not None:
        return dataclasses.replace(timeout, value=value, reason="distributed_timeout")

    if any(outcome.status != "safe" for outcome in outcomes):
        msg = "distributed worker returned an invalid status"
        raise DistributedError(msg)

    devices = []

    for outcome in outcomes:
        devices.extend(outcome.devices)

    return ProbeOutcome(
        status="safe",
        value=value,
        devices=tuple(devices),
        timing_seconds=_aggregate_timing(tuple(outcomes)),
        steps_completed=min(outcome.steps_completed for outcome in outcomes),
    )


def _aggregate_timing(outcomes: tuple[ProbeOutcome, ...]) -> tuple[float, ...]:
    lengths = {len(outcome.timing_seconds) for outcome in outcomes}

    if lengths == {0}:
        return ()

    if len(lengths) != 1:
        msg = "distributed timing sample counts differ across ranks"
        raise DistributedError(msg)

    count = lengths.pop()

    # Per-step max across ranks: distributed throughput is bounded by the slowest
    # rank, so the slowest rank's per-step time is what dominates real wall-clock.
    return tuple(
        max(outcome.timing_seconds[index] for outcome in outcomes)
        for index in range(count)
    )


def _ensure_parent_group() -> None:
    if not dist.is_available():
        msg = "torch.distributed is not available"
        raise DistributedError(msg)

    if dist.is_initialized():
        return

    dist.init_process_group(backend="gloo")


def _broadcast_candidate_port() -> int:
    values: list[object] = [_reserve_tcp_port()] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(values, src=0)

    return _broadcasted_port(values[0])


def _empty_object_list(count: int) -> list[object]:
    return [None for _ in range(count)]


def _selected_value_from_gathered(value: object) -> int:
    if type(value) is not int:
        msg = "parent rank selection must be an integer"
        raise DistributedError(msg)

    return value


def _outcome_from_gathered(value: object) -> ProbeOutcome:
    if not isinstance(value, Mapping):
        msg = "distributed worker outcome must be an object"
        raise DistributedError(msg)

    data = {}

    for key, item in value.items():
        if type(key) is not str:
            msg = "distributed worker outcome keys must be strings"
            raise DistributedError(msg)

        data[key] = item

    return ProbeOutcome.from_json(data)


def _broadcasted_port(value: object) -> int:
    if type(value) is not int:
        msg = "broadcast candidate port must be an integer"
        raise DistributedError(msg)

    return value


def _reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))

        return sock.getsockname()[1]


def _candidate_env(*, port: int, candidate_id: int) -> dict[str, str]:
    env = os.environ.copy()
    env["MASTER_PORT"] = str(port)
    env["AUTOBATCH_CANDIDATE_ID"] = str(candidate_id)
    env["AUTOBATCH_WORKER_DISTRIBUTED"] = "1"

    return env
