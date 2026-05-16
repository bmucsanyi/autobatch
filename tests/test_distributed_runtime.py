import queue
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import autobatch
import autobatch._probe as probe_module
import autobatch._staged as staged_module
from autobatch._cuda import DeviceRecord
from autobatch._errors import AutobatchError

_WORLD_SIZE = 2


def test_data_parallel_probe_collectives_select_same_value(tmp_path: Path) -> None:
    _require_gloo()

    results = _run_scenario("data_parallel", tmp_path)

    assert results == [(0, "ok", 2), (1, "ok", 2)]


def test_distributed_find_rejects_divergent_domains(tmp_path: Path) -> None:
    _require_gloo()

    results = _run_scenario("divergent_domain", tmp_path)

    assert results[0][1] == "error"
    assert results[1][1] == "error"
    assert results[0][2] == "DistributedError"
    assert results[1][2] == "DistributedError"


def test_distributed_find_rejects_plain_callable(tmp_path: Path) -> None:
    _require_gloo()

    results = _run_scenario("plain_callable", tmp_path)

    assert results[0][1] == "error"
    assert results[1][1] == "error"
    assert results[0][2] == "DistributedError"
    assert results[1][2] == "DistributedError"


def _run_scenario(scenario: str, tmp_path: Path) -> list[tuple[object, ...]]:
    rendezvous_path = tmp_path / f"{scenario}.dist"
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_distributed_rank,
            args=(rank, str(rendezvous_path), scenario, result_queue),
        )
        for rank in range(_WORLD_SIZE)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=20)

    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            msg = "distributed rank process hung"
            raise AssertionError(msg)

    for process in processes:
        if process.exitcode != 0:
            msg = f"distributed rank process exited with {process.exitcode}"
            raise AssertionError(msg)

    results = []

    for _ in processes:
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty as error:
            msg = "distributed rank process produced no result"
            raise AssertionError(msg) from error

    return sorted(results)


def _distributed_rank(
    rank: int,
    rendezvous_path: str,
    scenario: str,
    result_queue: Any,
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous_path}",
            rank=rank,
            world_size=_WORLD_SIZE,
        )
        _patch_probe_cuda()
        value = _run_rank_scenario(rank, scenario)
        result_queue.put((rank, "ok", value))
    except (AssertionError, AutobatchError, OSError, RuntimeError, ValueError) as error:
        result_queue.put((rank, "error", type(error).__name__, str(error)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_rank_scenario(rank: int, scenario: str) -> int:
    if scenario == "data_parallel":
        return _run_data_parallel_probe()

    if scenario == "divergent_domain":
        return _run_divergent_domain(rank)

    if scenario == "plain_callable":
        return _run_plain_callable()

    msg = f"unknown distributed test scenario: {scenario}"
    raise AssertionError(msg)


def _run_data_parallel_probe() -> int:
    tensor = torch.empty(1, dtype=torch.float32)

    def local(value: int) -> None:
        if dist.get_rank() == 0 and value > 2:
            msg = "CUDA out of memory"
            raise RuntimeError(msg)

    def sync(value: int, active: bool) -> None:
        payload = value if active else 0
        tensor.fill_(payload)
        dist.all_reduce(tensor)

    return autobatch.find(
        autobatch.StagedProbe([autobatch.ProbeStage(local, sync)]),
        values=[1, 2, 3],
        goal=autobatch.Goal.largest_safe(),
        cache_key=("distributed-runtime", "data-parallel"),
        warmup_steps=0,
        measure_steps=1,
        devices=[0],
    )


def _run_plain_callable() -> int:
    def probe(value: int) -> None:
        _ = value

    return autobatch.find(
        probe,
        values=[1, 2, 3],
        goal=autobatch.Goal.largest_safe(),
        cache_key=("distributed-runtime", "plain-callable"),
        warmup_steps=0,
        measure_steps=1,
        devices=[0],
    )


def _run_divergent_domain(rank: int) -> int:
    def local(value: int) -> None:
        _ = value

    def sync(value: int, active: bool) -> None:
        _ = value
        _ = active

    values = [1, 2] if rank == 0 else [1, 2, 3]

    return autobatch.find(
        autobatch.StagedProbe([autobatch.ProbeStage(local, sync)]),
        values=values,
        goal=autobatch.Goal.largest_safe(),
        cache_key=("distributed-runtime", "divergent-domain"),
        warmup_steps=0,
        measure_steps=1,
        devices=[0],
    )


def _patch_probe_cuda() -> None:
    def configure_devices(devices: list[int]) -> list[int]:
        return devices

    def no_op() -> None:
        return None

    def no_op_devices(devices: list[int]) -> None:
        _ = devices

    def read_memory_stats(devices: list[int]) -> tuple[DeviceRecord, ...]:
        _ = devices

        return (DeviceRecord("cuda:0", 10, 20),)

    probe_module.__dict__["configure_devices"] = configure_devices
    probe_module.torch.cuda.__dict__["empty_cache"] = no_op
    probe_module.__dict__["reset_peak_stats"] = no_op_devices
    probe_module.__dict__["synchronize_devices"] = no_op_devices
    probe_module.__dict__["read_memory_stats"] = read_memory_stats
    staged_module.__dict__["configure_devices"] = configure_devices
    staged_module.torch.cuda.__dict__["empty_cache"] = no_op
    staged_module.__dict__["reset_peak_stats"] = no_op_devices
    staged_module.__dict__["synchronize_devices"] = no_op_devices
    staged_module.__dict__["read_memory_stats"] = read_memory_stats


def _require_gloo() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed gloo is required")
