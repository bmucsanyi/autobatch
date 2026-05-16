import pytest
import torch

from autobatch._config import FindConfig, cache_key_identity, validate_config
from autobatch._distributed import (
    DistributedCache,
    TorchCollectives,
    _outcome_to_data,
    reduce_probe_outcome,
    validate_distributed_config,
    validate_distributed_selection,
)
from autobatch._errors import DistributedError
from autobatch._goals import Goal
from autobatch._probe import ProbeOutcome


class Collectives:
    def __init__(
        self,
        *,
        status_code: int,
        outcomes: tuple[ProbeOutcome, ...],
        timing_seconds: tuple[float, ...] = (),
        gathered: tuple[object, ...] = (),
    ) -> None:
        self.status_code = status_code
        self.outcomes = outcomes
        self.timing_seconds = timing_seconds
        self.status_inputs = []
        self.timing_inputs = []
        self.mapping_inputs = []
        self.int_inputs = []
        self.optional_int_inputs = []
        self.gathered = gathered

    def max_int(self, value: int) -> int:
        self.status_inputs.append(value)

        return self.status_code

    def max_float_tuple(self, values: tuple[float, ...]) -> tuple[float, ...]:
        self.timing_inputs.append(values)

        return self.timing_seconds

    def gather_int(self, value: int) -> tuple[int, ...]:
        self.int_inputs.append(value)

        return tuple(item for item in self.gathered if type(item) is int)

    def gather_optional_int(self, value: int | None) -> tuple[int | None, ...]:
        self.optional_int_inputs.append(value)

        return tuple(
            item if item is None or type(item) is int else None
            for item in self.gathered
        )

    def gather_mapping(self, value: object) -> tuple[dict[str, object], ...]:
        self.mapping_inputs.append(value)

        if isinstance(value, dict) and "status" in value:
            return tuple(_outcome_to_data(outcome) for outcome in self.outcomes)

        return tuple(
            require_mapping(item) for item in self.gathered if isinstance(item, dict)
        )


class Cache:
    def __init__(self, value: int | None) -> None:
        self.value = value
        self.writes = []

    def read_value(self) -> int | None:
        return self.value

    def write_value(self, value: int) -> None:
        self.writes.append(value)


def require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = "value must be a dict"
        raise TypeError(msg)

    result = {}

    for key, item in value.items():
        if not isinstance(key, str):
            msg = "key must be a string"
            raise TypeError(msg)

        result[key] = item

    return result


def control_group() -> object:
    return object()


def test_reduce_probe_outcome_uses_max_timing_when_all_ranks_are_safe() -> None:
    local = ProbeOutcome(
        status="safe",
        value=4,
        timing_seconds=(0.2, 0.3),
        steps_completed=2,
    )
    collectives = Collectives(
        status_code=0,
        outcomes=(local,),
        timing_seconds=(0.5, 0.4),
    )

    outcome = reduce_probe_outcome(
        value=4,
        local=local,
        collectives=collectives,
    )

    assert outcome.status == "safe"
    assert outcome.timing_seconds == (0.5, 0.4)
    assert collectives.status_inputs == [0]
    assert collectives.timing_inputs == [(0.2, 0.3)]


def test_reduce_probe_outcome_returns_unsafe_if_any_rank_is_unsafe() -> None:
    local = ProbeOutcome(status="safe", value=4, steps_completed=1)
    unsafe = ProbeOutcome(status="unsafe", value=4, reason="cuda_oom")
    collectives = Collectives(status_code=1, outcomes=(local, unsafe))

    outcome = reduce_probe_outcome(
        value=4,
        local=local,
        collectives=collectives,
    )

    assert outcome.status == "unsafe"
    assert outcome.reason == "cuda_oom"
    assert collectives.mapping_inputs == [_outcome_to_data(local)]


def test_reduce_probe_outcome_returns_failed_if_any_rank_failed() -> None:
    local = ProbeOutcome(status="safe", value=4, steps_completed=1)
    failed = ProbeOutcome(
        status="failed",
        value=4,
        reason="probe_exception",
        exception_message="bad operation",
    )
    collectives = Collectives(status_code=2, outcomes=(local, failed))

    outcome = reduce_probe_outcome(
        value=4,
        local=local,
        collectives=collectives,
    )

    assert outcome.status == "failed"
    assert outcome.exception_message == "bad operation"


def test_distributed_cache_uses_value_only_when_all_ranks_match() -> None:
    local_cache = Cache(8)
    collectives = Collectives(status_code=0, outcomes=(), gathered=(8, 8))
    cache = DistributedCache(local_cache, collectives)

    assert cache.read_value() == 8


def test_distributed_cache_rejects_divergent_rank_values() -> None:
    local_cache = Cache(8)
    collectives = Collectives(status_code=0, outcomes=(), gathered=(8, None))
    cache = DistributedCache(local_cache, collectives)

    with pytest.raises(DistributedError, match="cache values differ"):
        cache.read_value()


def noop_probe(value: int) -> None:
    _ = value


def make_config(values: tuple[int, ...]) -> FindConfig:
    return validate_config(
        probe=noop_probe,
        values=values,
        goal=Goal.largest_safe(),
        cache_key=("distributed-config", values),
        warmup_steps=1,
        measure_steps=1,
        devices=[0],
    )


def test_distributed_config_accepts_identical_rank_configuration() -> None:
    config = make_config((1, 2))
    collectives = Collectives(
        status_code=0,
        outcomes=(),
        gathered=(
            {
                "cache_key": cache_key_identity(("distributed-config", (1, 2))),
                "device_count": len(config.devices),
                "domain": list(config.domain.values),
                "goal": config.goal.to_json(),
                "measure_steps": config.measure_steps,
                "stage_count": 0,
                "warmup_steps": config.warmup_steps,
            },
        ),
    )

    validate_distributed_config(config, collectives)


def test_distributed_config_rejects_divergent_rank_configuration() -> None:
    config = make_config((1, 2))
    collectives = Collectives(
        status_code=0,
        outcomes=(),
        gathered=({"domain": [1, 2]}, {"domain": [1, 3]}),
    )

    with pytest.raises(DistributedError, match="distributed ranks"):
        validate_distributed_config(config, collectives)


def test_distributed_config_rejects_divergent_stage_count() -> None:
    config = make_config((1, 2))
    local_identity = {
        "cache_key": cache_key_identity(("distributed-config", (1, 2))),
        "device_count": len(config.devices),
        "domain": list(config.domain.values),
        "goal": config.goal.to_json(),
        "measure_steps": config.measure_steps,
        "stage_count": 0,
        "warmup_steps": config.warmup_steps,
    }
    peer_identity = {
        "cache_key": cache_key_identity(("distributed-config", (1, 2))),
        "device_count": len(config.devices),
        "domain": list(config.domain.values),
        "goal": config.goal.to_json(),
        "measure_steps": config.measure_steps,
        "stage_count": 1,
        "warmup_steps": config.warmup_steps,
    }
    collectives = Collectives(
        status_code=0,
        outcomes=(),
        gathered=(local_identity, peer_identity),
    )

    with pytest.raises(DistributedError, match="distributed ranks"):
        validate_distributed_config(config, collectives)


def test_distributed_config_rejects_bool_int_cache_key_mismatch() -> None:
    config = validate_config(
        probe=noop_probe,
        values=(1, 2),
        goal=Goal.largest_safe(),
        cache_key=True,
        warmup_steps=1,
        measure_steps=1,
        devices=[0],
    )
    local_identity = {
        "cache_key": cache_key_identity(True),
        "device_count": len(config.devices),
        "domain": list(config.domain.values),
        "goal": config.goal.to_json(),
        "measure_steps": config.measure_steps,
        "stage_count": 0,
        "warmup_steps": config.warmup_steps,
    }
    peer_identity = {
        "cache_key": cache_key_identity(1),
        "device_count": len(config.devices),
        "domain": list(config.domain.values),
        "goal": config.goal.to_json(),
        "measure_steps": config.measure_steps,
        "stage_count": 0,
        "warmup_steps": config.warmup_steps,
    }
    collectives = Collectives(
        status_code=0,
        outcomes=(),
        gathered=(local_identity, peer_identity),
    )

    with pytest.raises(DistributedError, match="distributed ranks"):
        validate_distributed_config(config, collectives)


def test_distributed_selection_rejects_divergent_selected_values() -> None:
    collectives = Collectives(status_code=0, outcomes=(), gathered=(2, 3))

    with pytest.raises(DistributedError, match="selected different values"):
        validate_distributed_selection(2, collectives)


def test_torch_collectives_close_destroys_control_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = control_group()
    destroyed = []
    collectives = object.__new__(TorchCollectives)
    collectives.group = group

    def destroy_process_group(selected_group: object) -> None:
        destroyed.append(selected_group)

    monkeypatch.setattr(
        "autobatch._distributed.dist.destroy_process_group",
        destroy_process_group,
    )

    collectives.close()

    assert destroyed == [group]
    assert collectives.group is None


def test_torch_collectives_control_tensors_ignore_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collectives = object.__new__(TorchCollectives)
    collectives.group = control_group()
    devices = []

    def all_reduce(tensor: torch.Tensor, *, op: object, group: object) -> None:
        _ = op
        _ = group
        devices.append(tensor.device.type)

    def all_gather(
        gathered: list[torch.Tensor],
        tensor: torch.Tensor,
        *,
        group: object,
    ) -> None:
        _ = group
        devices.append(tensor.device.type)

        for item in gathered:
            devices.append(item.device.type)
            item.copy_(tensor)

    monkeypatch.setattr(
        "autobatch._distributed.dist.get_world_size",
        lambda _: 2,
    )
    monkeypatch.setattr("autobatch._distributed.dist.all_reduce", all_reduce)
    monkeypatch.setattr("autobatch._distributed.dist.all_gather", all_gather)
    torch.set_default_device("meta")

    try:
        assert collectives.max_int(2) == 2
        assert collectives.gather_int(3) == (3, 3)
        assert collectives.gather_optional_int(4) == (4, 4)
    finally:
        torch.set_default_device("cpu")

    assert set(devices) == {"cpu"}
