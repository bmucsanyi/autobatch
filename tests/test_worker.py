import pytest

from autobatch._protocol import DeviceRecord, ProbeRequest
from autobatch._worker import evaluate_request


def request(*, value: int, need_timing: bool) -> ProbeRequest:
    return ProbeRequest(
        workload="tests.fixtures.probes:step",
        kwargs={},
        value=value,
        warmup_steps=2,
        measure_steps=3,
        reserve_fraction=0.05,
        reserve_bytes=0,
        devices=[0],
        need_timing=need_timing,
        rank=None,
        world_size=None,
        local_rank=None,
        master_addr=None,
        master_port=None,
        candidate_id=None,
    )


def test_worker_reports_cuda_unavailable_as_workload_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_devices(_: list[int]) -> list[int]:
        msg = "CUDA illegal memory access"
        raise RuntimeError(msg)

    monkeypatch.setattr("autobatch._worker.select_devices", no_devices)
    outcome = evaluate_request(request(value=1, need_timing=False))

    assert outcome.status == "failed"


def test_worker_runs_declared_steps_and_returns_safe_outcome(
    monkeypatch: pytest.MonkeyPatch,
    probe_calls: list[int],
) -> None:
    records = []

    def devices(_: list[int]) -> list[int]:
        return [0]

    def no_op() -> None:
        return None

    def no_op_devices(_: list[int]) -> None:
        return None

    def memory_records(
        _devices: list[int],
        *,
        reserve_fraction: float,
        reserve_bytes: int,
    ) -> tuple[DeviceRecord, ...]:
        _ = reserve_fraction
        _ = reserve_bytes
        record = DeviceRecord("cuda:0", 10, 20, 30)
        records.append(record)

        return (record,)

    monkeypatch.setattr("autobatch._worker.select_devices", devices)
    monkeypatch.setattr("autobatch._worker.torch.cuda.empty_cache", no_op)
    monkeypatch.setattr("autobatch._worker.reset_peak_stats", no_op_devices)
    monkeypatch.setattr("autobatch._worker.synchronize_devices", no_op_devices)
    monkeypatch.setattr("autobatch._worker.read_memory_stats", memory_records)

    outcome = evaluate_request(request(value=7, need_timing=False))

    assert outcome.status == "safe"
    assert probe_calls == [7, 7, 7, 7, 7]
    assert outcome.steps_completed == 5
    assert outcome.devices[0].budget_bytes == 30
    assert records


def test_worker_reports_memory_budget_violation(
    monkeypatch: pytest.MonkeyPatch,
    probe_calls: list[int],
) -> None:
    def devices(_: list[int]) -> list[int]:
        return [0]

    def no_op() -> None:
        return None

    def no_op_devices(_: list[int]) -> None:
        return None

    def memory_records(
        _devices: list[int],
        *,
        reserve_fraction: float,
        reserve_bytes: int,
    ) -> tuple[DeviceRecord, ...]:
        _ = reserve_fraction
        _ = reserve_bytes

        return (DeviceRecord("cuda:0", 10, 40, 30),)

    monkeypatch.setattr("autobatch._worker.select_devices", devices)
    monkeypatch.setattr("autobatch._worker.torch.cuda.empty_cache", no_op)
    monkeypatch.setattr("autobatch._worker.reset_peak_stats", no_op_devices)
    monkeypatch.setattr("autobatch._worker.synchronize_devices", no_op_devices)
    monkeypatch.setattr("autobatch._worker.read_memory_stats", memory_records)

    outcome = evaluate_request(request(value=9, need_timing=False))

    assert outcome.status == "unsafe"
    assert outcome.reason == "memory_budget_exceeded"
    assert probe_calls == [9, 9, 9, 9, 9]
    assert outcome.devices[0].peak_reserved_bytes == 40
