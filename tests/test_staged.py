import pytest

from autobatch._config import FindConfig, validate_config
from autobatch._cuda import DeviceRecord
from autobatch._goals import Goal
from autobatch._probe import InProcessProbeRunner
from autobatch._staged import DistributedStagedProbeRunner
from autobatch._types import ProbeStage, StagedProbe


class Collectives:
    def __init__(self, max_int_values: tuple[int, ...]) -> None:
        self.max_int_values = list(max_int_values)
        self.max_int_inputs = []
        self.timing_inputs = []
        self.int_inputs = []
        self.optional_int_inputs = []
        self.mapping_inputs = []

    def max_int(self, value: int) -> int:
        self.max_int_inputs.append(value)

        return self.max_int_values.pop(0)

    def max_float_tuple(self, values: tuple[float, ...]) -> tuple[float, ...]:
        self.timing_inputs.append(values)

        return values

    def gather_int(self, value: int) -> tuple[int, ...]:
        self.int_inputs.append(value)

        return (value,)

    def gather_optional_int(self, value: int | None) -> tuple[int | None, ...]:
        self.optional_int_inputs.append(value)

        return (value,)

    def gather_mapping(self, value: object) -> tuple[dict[str, object], ...]:
        self.mapping_inputs.append(value)

        if not isinstance(value, dict):
            msg = "gathered value must be a dict"
            raise TypeError(msg)

        result = {}

        for key, item in value.items():
            if not isinstance(key, str):
                msg = "gathered key must be a string"
                raise TypeError(msg)

            result[key] = item

        return (result,)


def patch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    records = (DeviceRecord("cuda:0", 10, 20),)

    def configure_devices(devices: list[int]) -> list[int]:
        return devices

    def no_op() -> None:
        return None

    def no_op_devices(devices: list[int]) -> None:
        _ = devices

    def memory_records(devices: list[int]) -> tuple[DeviceRecord, ...]:
        _ = devices

        return records

    monkeypatch.setattr("autobatch._probe.configure_devices", configure_devices)
    monkeypatch.setattr("autobatch._probe.torch.cuda.empty_cache", no_op)
    monkeypatch.setattr("autobatch._probe.reset_peak_stats", no_op_devices)
    monkeypatch.setattr("autobatch._probe.synchronize_devices", no_op_devices)
    monkeypatch.setattr("autobatch._probe.read_memory_stats", memory_records)
    monkeypatch.setattr("autobatch._staged.configure_devices", configure_devices)
    monkeypatch.setattr("autobatch._staged.torch.cuda.empty_cache", no_op)
    monkeypatch.setattr("autobatch._staged.reset_peak_stats", no_op_devices)
    monkeypatch.setattr("autobatch._staged.synchronize_devices", no_op_devices)
    monkeypatch.setattr("autobatch._staged.read_memory_stats", memory_records)


def make_config(probe: StagedProbe) -> FindConfig:
    return validate_config(
        probe=probe,
        values=[1, 2, 3],
        goal=Goal.largest_safe(),
        cache_key=("staged", id(probe)),
        warmup_steps=0,
        measure_steps=1,
        devices=[0],
    )


def test_staged_probe_runs_sync_in_single_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    patch_cuda(monkeypatch)

    def local(value: int) -> None:
        events.append(("local", value))

    def sync(value: int, active: bool) -> None:
        events.append(("sync", value, active))

    runner = InProcessProbeRunner(make_config(StagedProbe([ProbeStage(local, sync)])))
    outcome = runner.probe(2, timed=False)

    assert outcome.status == "safe"
    assert events == [("local", 2), ("sync", 2, True)]


def test_distributed_staged_probe_drains_remaining_syncs_after_peer_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    patch_cuda(monkeypatch)

    def first_local(value: int) -> None:
        events.append(("first_local", value))

    def first_sync(value: int, active: bool) -> None:
        events.append(("first_sync", value, active))

    def second_local(value: int) -> None:
        events.append(("second_local", value))

    def second_sync(value: int, active: bool) -> None:
        events.append(("second_sync", value, active))

    staged = StagedProbe([
        ProbeStage(first_local, first_sync),
        ProbeStage(second_local, second_sync),
    ])
    runner = DistributedStagedProbeRunner(
        make_config(staged),
        Collectives((1, 0, 0, 0, 1)),
    )
    outcome = runner.probe(3, timed=False)

    assert outcome.status == "unsafe"
    assert events == [
        ("first_local", 3),
        ("first_sync", 3, False),
        ("second_sync", 3, False),
    ]


def test_distributed_staged_probe_drains_remaining_syncs_after_sync_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    patch_cuda(monkeypatch)

    def first_local(value: int) -> None:
        events.append(("first_local", value))

    def first_sync(value: int, active: bool) -> None:
        events.append(("first_sync", value, active))
        msg = "bad sync"
        raise RuntimeError(msg)

    def second_local(value: int) -> None:
        events.append(("second_local", value))

    def second_sync(value: int, active: bool) -> None:
        events.append(("second_sync", value, active))

    staged = StagedProbe([
        ProbeStage(first_local, first_sync),
        ProbeStage(second_local, second_sync),
    ])
    runner = DistributedStagedProbeRunner(
        make_config(staged),
        Collectives((0, 2, 0, 0, 2)),
    )
    outcome = runner.probe(3, timed=False)

    assert outcome.status == "failed"
    assert outcome.exception_message == "bad sync"
    assert events == [
        ("first_local", 3),
        ("first_sync", 3, True),
        ("second_sync", 3, False),
    ]
