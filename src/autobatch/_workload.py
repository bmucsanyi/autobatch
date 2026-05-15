import importlib
import os
from collections.abc import Callable, Mapping

from autobatch._errors import InvalidConfigurationError, WorkloadError
from autobatch._timing import run_step


def load_workload(path: str) -> Callable[..., object]:
    module_name, attribute_name = split_import_path(path)
    module = importlib.import_module(module_name)
    target = module

    for part in attribute_name.split("."):
        target = getattr(target, part)

    if not callable(target):
        msg = "workload target must be callable"
        raise WorkloadError(msg)

    return target


def split_import_path(path: str) -> tuple[str, str]:
    module_name, separator, attribute_name = path.partition(":")

    if separator != ":" or not module_name or not attribute_name:
        msg = "workload import path must use module:attribute"
        raise InvalidConfigurationError(msg)

    return module_name, attribute_name


def run_workload(
    workload: Callable[..., object],
    value: int,
    *,
    kwargs: Mapping[str, object],
    warmup_steps: int,
    measure_steps: int,
    need_timing: bool,
    devices: list[int],
) -> tuple[float, ...]:
    def step() -> object:
        return workload(value, **kwargs)

    samples = []
    single_device_cuda_events = len(devices) == 1 and not _distributed_worker_enabled()

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
            timed=need_timing,
            single_device_cuda_events=single_device_cuda_events,
        )

        if sample is not None:
            samples.append(sample)

    return tuple(samples)


def _distributed_worker_enabled() -> bool:
    return os.environ.get("AUTOBATCH_WORKER_DISTRIBUTED") == "1"
