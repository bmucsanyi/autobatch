import time
from collections.abc import Callable, Sequence

import torch

from autobatch._cuda import synchronize_devices
from autobatch._errors import WorkloadError


def run_step(
    step: Callable[[], object],
    *,
    devices: Sequence[int],
    timed: bool,
    single_device_cuda_events: bool,
) -> float | None:
    if not timed:
        step()

        return None

    if single_device_cuda_events:
        return _run_step_with_cuda_event(step, devices[0])

    return _run_step_with_host_clock(step, devices)


def median_seconds(samples: Sequence[float]) -> float:
    if len(samples) == 0:
        msg = "timing samples must be nonempty"
        raise WorkloadError(msg)

    ordered = sorted(samples)
    middle = len(ordered) // 2

    if len(ordered) % 2 == 1:
        return ordered[middle]

    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _run_step_with_cuda_event(step: Callable[[], object], device: int) -> float:
    synchronize_devices([device])
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    step()
    end.record()
    torch.cuda.synchronize(device)

    return start.elapsed_time(end) / 1000.0


def _run_step_with_host_clock(
    step: Callable[[], object], devices: Sequence[int]
) -> float:
    synchronize_devices(devices)
    start = time.perf_counter()
    step()
    synchronize_devices(devices)
    end = time.perf_counter()

    return end - start
