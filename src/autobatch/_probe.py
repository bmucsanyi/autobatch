from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from autobatch._config import FindConfig
from autobatch._cuda import (
    DeviceRecord,
    configure_devices,
    read_memory_stats,
    reset_peak_stats,
    synchronize_devices,
)
from autobatch._errors import ProbeError
from autobatch._oom import is_oom_error
from autobatch._timing import run_step
from autobatch._types import StagedProbe

_PROBE_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    MemoryError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(slots=True)
class ProbeOutcome:
    status: str
    value: int
    devices: tuple[DeviceRecord, ...] = ()
    timing_seconds: tuple[float, ...] = ()
    steps_completed: int = 0
    reason: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None

    def __post_init__(self) -> None:
        self.devices = tuple(self.devices)
        self.timing_seconds = tuple(self.timing_seconds)

        if self.status not in {"safe", "unsafe", "failed"}:
            msg = "probe status must be safe, unsafe, or failed"
            raise ProbeError(msg)

        if self.status == "safe" and self.reason is not None:
            msg = "safe outcome cannot have a reason"
            raise ProbeError(msg)

        if self.status != "safe" and self.reason is None:
            msg = "non-safe outcome must have a reason"
            raise ProbeError(msg)


class InProcessProbeRunner:
    def __init__(self, config: FindConfig) -> None:
        self.config = config

    def probe(self, value: int, *, timed: bool) -> ProbeOutcome:
        try:
            return self._probe_or_raise(value, timed=timed)
        except _PROBE_EXCEPTIONS as error:
            if is_oom_error(error):
                torch.cuda.empty_cache()

                return ProbeOutcome(
                    status="unsafe",
                    value=value,
                    reason="cuda_oom",
                    exception_type=type(error).__name__,
                    exception_message=str(error),
                )

            return ProbeOutcome(
                status="failed",
                value=value,
                reason="probe_exception",
                exception_type=type(error).__name__,
                exception_message=str(error),
            )

    def _probe_or_raise(self, value: int, *, timed: bool) -> ProbeOutcome:
        devices = configure_devices(self.config.devices)
        torch.cuda.empty_cache()
        reset_peak_stats(devices)
        timing_seconds = run_probe_steps(
            self.config.probe,
            value,
            warmup_steps=self.config.warmup_steps,
            measure_steps=self.config.measure_steps,
            devices=devices,
            timed=timed,
        )
        synchronize_devices(devices)
        records = read_memory_stats(devices)
        steps_completed = self.config.warmup_steps + self.config.measure_steps

        return ProbeOutcome(
            status="safe",
            value=value,
            devices=records,
            timing_seconds=timing_seconds,
            steps_completed=steps_completed,
        )


def run_probe_steps(
    probe: Callable[[int], None] | StagedProbe,
    value: int,
    *,
    warmup_steps: int,
    measure_steps: int,
    devices: Sequence[int],
    timed: bool,
) -> tuple[float, ...]:
    def step() -> None:
        if isinstance(probe, StagedProbe):
            for stage in probe.stages:
                stage.local(value)
                stage.sync(value, True)

            return

        probe(value)

    samples = []
    single_device_cuda_events = len(devices) == 1

    for _ in range(warmup_steps):
        run_step(
            step,
            devices=devices,
            timed=False,
            single_device_cuda_events=False,
        )

    for _ in range(measure_steps):
        sample = run_step(
            step,
            devices=devices,
            timed=timed,
            single_device_cuda_events=single_device_cuda_events,
        )

        if sample is not None:
            samples.append(sample)

    return tuple(samples)
