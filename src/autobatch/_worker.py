import argparse
import sys
from pathlib import Path

import torch

from autobatch._cuda import (
    read_memory_stats,
    records_fit_budget,
    reset_peak_stats,
    select_devices,
    synchronize_devices,
)
from autobatch._dist_worker import (
    setup_candidate_process_group,
    teardown_candidate_process_group,
)
from autobatch._errors import InvalidConfigurationError, WorkloadError
from autobatch._oom import is_oom_error
from autobatch._protocol import ProbeOutcome, ProbeRequest, read_json, write_json_atomic
from autobatch._workload import load_workload, run_workload


def evaluate_request(request: ProbeRequest) -> ProbeOutcome:
    try:
        return _evaluate_request_or_raise(request)
    except (
        AttributeError,
        ImportError,
        InvalidConfigurationError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        WorkloadError,
    ) as error:
        if is_oom_error(error):
            return ProbeOutcome(
                status="unsafe",
                value=request.value,
                reason="cuda_oom",
                exception_type=type(error).__name__,
                exception_message=str(error),
                rank=request.rank,
            )

        return ProbeOutcome(
            status="failed",
            value=request.value,
            reason="workload_exception",
            exception_type=type(error).__name__,
            exception_message=str(error),
            rank=request.rank,
        )


def _evaluate_request_or_raise(request: ProbeRequest) -> ProbeOutcome:
    created = setup_candidate_process_group(request)

    try:
        return _run_user_probe(request)
    finally:
        teardown_candidate_process_group(created)


def _run_user_probe(request: ProbeRequest) -> ProbeOutcome:
    devices = select_devices(request.devices)
    torch.cuda.empty_cache()
    reset_peak_stats(devices)
    workload = load_workload(request.workload)
    timing_seconds = run_workload(
        workload,
        request.value,
        kwargs=request.kwargs,
        warmup_steps=request.warmup_steps,
        measure_steps=request.measure_steps,
        need_timing=request.need_timing,
        devices=devices,
    )
    synchronize_devices(devices)
    records = read_memory_stats(
        devices,
        reserve_fraction=request.reserve_fraction,
        reserve_bytes=request.reserve_bytes,
    )

    if not records_fit_budget(records):
        return ProbeOutcome(
            status="unsafe",
            value=request.value,
            reason="memory_budget_exceeded",
            devices=records,
            timing_seconds=timing_seconds,
            steps_completed=request.warmup_steps + request.measure_steps,
            rank=request.rank,
        )

    return ProbeOutcome(
        status="safe",
        value=request.value,
        devices=records,
        timing_seconds=timing_seconds,
        steps_completed=request.warmup_steps + request.measure_steps,
        rank=request.rank,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m autobatch._worker")
    parser.add_argument("request_path")
    parser.add_argument("outcome_path")
    args = parser.parse_args(argv)
    request = ProbeRequest.from_json(read_json(Path(args.request_path)))
    outcome = evaluate_request(request)
    write_json_atomic(Path(args.outcome_path), outcome.to_json())

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
