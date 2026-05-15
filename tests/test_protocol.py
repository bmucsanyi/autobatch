import pytest

from autobatch._errors import WorkloadError
from autobatch._protocol import (
    DeviceRecord,
    ProbeOutcome,
    ProbeRequest,
)


def test_probe_request_round_trips() -> None:
    request = ProbeRequest(
        workload="tests.fixtures.probes:step",
        kwargs={"shape": 4},
        value=8,
        warmup_steps=1,
        measure_steps=2,
        reserve_fraction=0.05,
        reserve_bytes=0,
        devices=[0],
        need_timing=True,
        rank=None,
        world_size=None,
        local_rank=None,
        master_addr=None,
        master_port=None,
        candidate_id=None,
    )

    parsed = ProbeRequest.from_json(request.to_json())

    assert parsed == request


def test_probe_outcome_round_trips() -> None:
    outcome = ProbeOutcome(
        status="safe",
        value=4,
        devices=(DeviceRecord("cuda:0", 10, 12, 20),),
        steps_completed=3,
        timing_seconds=(0.2, 0.1),
    )

    parsed = ProbeOutcome.from_json(outcome.to_json())

    assert parsed == outcome


def test_probe_outcome_rejects_bad_status() -> None:
    with pytest.raises(WorkloadError):
        ProbeOutcome.from_json({
            "status": "bad",
            "value": 1,
            "devices": [],
            "timing_seconds": [],
            "steps_completed": 0,
            "reason": None,
            "exception_type": None,
            "exception_message": None,
            "failure_class": None,
            "stderr_path": None,
            "rank": None,
        })


def test_probe_request_rejects_timed_alias() -> None:
    data = {
        "workload": "tests.fixtures.probes:step",
        "kwargs": {},
        "value": 8,
        "warmup_steps": 1,
        "measure_steps": 2,
        "reserve_fraction": 0.05,
        "reserve_bytes": 0,
        "devices": [0],
        "timed": True,
        "rank": None,
        "world_size": None,
        "local_rank": None,
        "master_addr": None,
        "master_port": None,
        "candidate_id": None,
    }

    with pytest.raises(WorkloadError):
        ProbeRequest.from_json(data)


def test_timeout_round_trips() -> None:
    outcome = ProbeOutcome(status="timeout", value=5, reason="timeout")

    assert ProbeOutcome.from_json(outcome.to_json()).status == "timeout"


def test_unsafe_outcome_keeps_device_records() -> None:
    outcome = ProbeOutcome(
        status="unsafe",
        value=4,
        reason="memory_budget_exceeded",
        devices=(DeviceRecord("cuda:0", 10, 12, 11),),
    )

    parsed = ProbeOutcome.from_json(outcome.to_json())

    assert parsed.devices == outcome.devices
