from collections.abc import Callable, Sequence

import pytest

from autobatch._config import validate_config
from autobatch._cuda import DeviceRecord
from autobatch._goals import Goal
from autobatch._probe import InProcessProbeRunner


def make_runner(
    probe: Callable[[int], None], *, warmup_steps: int, measure_steps: int
) -> InProcessProbeRunner:
    config = validate_config(
        probe=probe,
        values=[1, 2, 3],
        goal=Goal.largest_safe(),
        cache_key=("probe-runner", id(probe), warmup_steps, measure_steps),
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        devices=[0],
    )

    return InProcessProbeRunner(config)


def patch_cuda(
    monkeypatch: pytest.MonkeyPatch,
    records: tuple[DeviceRecord, ...],
) -> None:
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


def test_runner_runs_declared_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    records = (DeviceRecord("cuda:0", 10, 20),)
    patch_cuda(monkeypatch, records)

    def probe(value: int) -> None:
        calls.append(value)

    runner = make_runner(probe, warmup_steps=2, measure_steps=3)
    outcome = runner.probe(7, timed=False)

    assert outcome.status == "safe"
    assert calls == [7, 7, 7, 7, 7]
    assert outcome.steps_completed == 5
    assert outcome.devices == records


def test_runner_classifies_oom_as_unsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    records = (DeviceRecord("cuda:0", 10, 20),)
    patch_cuda(monkeypatch, records)

    def probe(value: int) -> None:
        _ = value
        msg = "CUDA out of memory"
        raise RuntimeError(msg)

    runner = make_runner(probe, warmup_steps=0, measure_steps=1)
    outcome = runner.probe(4, timed=False)

    assert outcome.status == "unsafe"
    assert outcome.reason == "cuda_oom"


def test_runner_reports_non_oom_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (DeviceRecord("cuda:0", 10, 20),)
    patch_cuda(monkeypatch, records)

    def probe(value: int) -> None:
        _ = value
        msg = "bad operation"
        raise RuntimeError(msg)

    runner = make_runner(probe, warmup_steps=0, measure_steps=1)
    outcome = runner.probe(4, timed=False)

    assert outcome.status == "failed"
    assert outcome.exception_message == "bad operation"


def test_runner_records_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    records = (DeviceRecord("cuda:0", 10, 20),)
    patch_cuda(monkeypatch, records)

    def probe(value: int) -> None:
        _ = value

    def run_step(
        step: Callable[[], None],
        *,
        devices: Sequence[int],
        timed: bool,
        single_device_cuda_events: bool,
    ) -> float | None:
        _ = devices
        _ = single_device_cuda_events
        step()

        if timed:
            return 0.25

        return None

    monkeypatch.setattr("autobatch._probe.run_step", run_step)
    runner = make_runner(probe, warmup_steps=1, measure_steps=2)
    outcome = runner.probe(4, timed=True)

    assert outcome.status == "safe"
    assert outcome.timing_seconds == (0.25, 0.25)
