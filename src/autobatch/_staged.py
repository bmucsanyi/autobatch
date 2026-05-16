from collections.abc import Callable
from dataclasses import dataclass

import torch

from autobatch._config import FindConfig
from autobatch._cuda import (
    configure_devices,
    read_memory_stats,
    reset_peak_stats,
    synchronize_devices,
)
from autobatch._distributed import _Collectives, reduce_probe_outcome
from autobatch._errors import ProbeError
from autobatch._oom import is_oom_error
from autobatch._probe import ProbeOutcome
from autobatch._timing import run_step
from autobatch._types import StagedProbe

_STAGE_SAFE = 0
_STAGE_UNSAFE = 1
_STAGE_FAILED = 2

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
class _StageResult:
    status: str
    reason: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None


class DistributedStagedProbeRunner:
    def __init__(self, config: FindConfig, collectives: _Collectives) -> None:
        if not isinstance(config.probe, StagedProbe):
            msg = "distributed staged runner requires a staged probe"
            raise ProbeError(msg)

        self.config = config
        self.staged_probe = config.probe
        self.collectives = collectives

    def probe(self, value: int, *, timed: bool) -> ProbeOutcome:
        local = self._local_probe(value, timed=timed)

        return reduce_probe_outcome(
            value=value,
            local=local,
            collectives=self.collectives,
        )

    def _local_probe(self, value: int, *, timed: bool) -> ProbeOutcome:
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
        timing_seconds = []

        for _ in range(self.config.warmup_steps):
            result = self._run_once(value)

            if result.status != "safe":
                return _outcome_from_stage_result(value, result)

        for _ in range(self.config.measure_steps):
            result = _StageResult(status="safe")

            def step() -> None:
                nonlocal result
                result = self._run_once(value)

            sample = run_step(
                step,
                devices=devices,
                timed=timed,
                single_device_cuda_events=False,
            )

            if result.status != "safe":
                return _outcome_from_stage_result(value, result)

            if sample is not None:
                timing_seconds.append(sample)

        synchronize_devices(devices)
        records = read_memory_stats(devices)
        steps_completed = self.config.warmup_steps + self.config.measure_steps

        return ProbeOutcome(
            status="safe",
            value=value,
            devices=records,
            timing_seconds=tuple(timing_seconds),
            steps_completed=steps_completed,
        )

    def _run_once(self, value: int) -> _StageResult:
        active = True
        result = _StageResult(status="safe")

        for stage in self.staged_probe.stages:
            local_code = _STAGE_SAFE
            local_result = _StageResult(status="safe")

            if active:
                local_result = _run_stage_local(stage.local, value)
                local_code = _code_for_result(local_result)

            reduced_code = self.collectives.max_int(local_code)

            if reduced_code != _STAGE_SAFE:
                active = False

                if result.status == "safe":
                    result = _result_for_reduced_code(
                        reduced_code,
                        local_result=local_result,
                    )

            sync_result = _run_stage_sync(stage.sync, value, active)
            sync_code = _code_for_result(sync_result)
            reduced_sync_code = self.collectives.max_int(sync_code)

            if reduced_sync_code != _STAGE_SAFE:
                active = False

                if result.status == "safe":
                    result = _result_for_reduced_code(
                        reduced_sync_code,
                        local_result=sync_result,
                    )

        return result


def _run_stage_local(local: Callable[[int], None], value: int) -> _StageResult:
    try:
        local(value)
    except _PROBE_EXCEPTIONS as error:
        if is_oom_error(error):
            torch.cuda.empty_cache()

            return _StageResult(
                status="unsafe",
                reason="cuda_oom",
                exception_type=type(error).__name__,
                exception_message=str(error),
            )

        return _StageResult(
            status="failed",
            reason="probe_exception",
            exception_type=type(error).__name__,
            exception_message=str(error),
        )

    return _StageResult(status="safe")


def _run_stage_sync(
    sync: Callable[[int, bool], None],
    value: int,
    active: bool,
) -> _StageResult:
    try:
        sync(value, active)
    except _PROBE_EXCEPTIONS as error:
        if is_oom_error(error):
            torch.cuda.empty_cache()

            return _StageResult(
                status="unsafe",
                reason="cuda_oom",
                exception_type=type(error).__name__,
                exception_message=str(error),
            )

        return _StageResult(
            status="failed",
            reason="probe_exception",
            exception_type=type(error).__name__,
            exception_message=str(error),
        )

    return _StageResult(status="safe")


def _code_for_result(result: _StageResult) -> int:
    if result.status == "safe":
        return _STAGE_SAFE

    if result.status == "unsafe":
        return _STAGE_UNSAFE

    if result.status == "failed":
        return _STAGE_FAILED

    msg = "staged probe result status is invalid"
    raise ProbeError(msg)


def _result_for_reduced_code(
    code: int,
    *,
    local_result: _StageResult,
) -> _StageResult:
    if local_result.status != "safe":
        return local_result

    if code == _STAGE_UNSAFE:
        return _StageResult(status="unsafe", reason="distributed_peer_unsafe")

    if code == _STAGE_FAILED:
        return _StageResult(status="failed", reason="distributed_peer_failed")

    msg = "staged probe reduced status is invalid"
    raise ProbeError(msg)


def _outcome_from_stage_result(value: int, result: _StageResult) -> ProbeOutcome:
    return ProbeOutcome(
        status=result.status,
        value=value,
        reason=result.reason,
        exception_type=result.exception_type,
        exception_message=result.exception_message,
    )
