import os
import signal
import subprocess  # noqa: S404
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from autobatch._config import FindConfig
from autobatch._errors import WorkloadError
from autobatch._protocol import ProbeOutcome, ProbeRequest, read_json, write_json_atomic


class WorkerProcess:
    def __init__(
        self,
        *,
        request: ProbeRequest,
        env: Mapping[str, str] | None,
    ) -> None:
        self.request = request
        self.env = env
        self.directory = tempfile.TemporaryDirectory(prefix="autobatch-worker-")
        self.root = Path(self.directory.name)
        self.request_path = self.root / "request.json"
        self.outcome_path = self.root / "outcome.json"
        self.stderr_path = self.root / "stderr.txt"
        self.process = None

    def start(self) -> None:
        write_json_atomic(self.request_path, self.request.to_json())
        command = [
            sys.executable,
            "-m",
            "autobatch._worker",
            str(self.request_path),
            str(self.outcome_path),
        ]

        with self.stderr_path.open("wb") as stderr_file:
            if os.name == "nt":
                self.process = subprocess.Popen(  # noqa: S603
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    env=_worker_environment(self.env),
                )
            else:
                self.process = subprocess.Popen(  # noqa: S603
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    env=_worker_environment(self.env),
                    preexec_fn=os.setsid,  # noqa: PLW1509
                )

    def wait(self, timeout_s: float) -> ProbeOutcome:
        if self.process is None:
            msg = "worker process was not started"
            raise WorkloadError(msg)

        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.kill()

            return ProbeOutcome(
                status="timeout",
                value=self.request.value,
                reason="timeout",
                stderr_path=str(self.stderr_path),
            )

        if self.process.returncode != 0:
            return ProbeOutcome(
                status="failed",
                value=self.request.value,
                reason="worker_exit",
                exception_message=_read_stderr(self.stderr_path),
                stderr_path=str(self.stderr_path),
            )

        if not self.outcome_path.exists():
            return ProbeOutcome(
                status="failed",
                value=self.request.value,
                reason="worker_protocol_violation",
                exception_message=_read_stderr(self.stderr_path),
                stderr_path=str(self.stderr_path),
            )

        return ProbeOutcome.from_json(read_json(self.outcome_path))

    def kill(self) -> None:
        if self.process is None:
            return

        if self.process.poll() is not None:
            return

        if os.name == "nt":
            self.process.kill()
            self.process.wait()

            return

        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return

        self.process.wait()

    def close(self) -> None:
        self.directory.cleanup()


class SubprocessProbeRunner:
    def __init__(self, config: FindConfig) -> None:
        self.config = config

    def probe(self, value: int, *, timed: bool) -> ProbeOutcome:
        request = make_worker_request(self.config, value, timed=timed)

        return run_worker_process(request, timeout_s=self.config.timeout_s, env=None)


def run_worker_process(
    request: ProbeRequest,
    timeout_s: float,
    *,
    env: Mapping[str, str] | None,
) -> ProbeOutcome:
    worker = WorkerProcess(request=request, env=env)

    try:
        worker.start()

        return worker.wait(timeout_s)
    finally:
        worker.kill()
        worker.close()


def make_worker_request(config: FindConfig, value: int, *, timed: bool) -> ProbeRequest:
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")
    master_port = os.environ.get("MASTER_PORT")

    return ProbeRequest(
        workload=config.workload,
        value=value,
        kwargs=config.kwargs,
        warmup_steps=config.warmup_steps,
        measure_steps=config.measure_steps,
        need_timing=timed,
        reserve_fraction=config.reserve_fraction,
        reserve_bytes=config.reserve_bytes,
        devices=config.devices,
        rank=int(rank) if rank is not None else None,
        world_size=int(world_size) if world_size is not None else None,
        local_rank=int(local_rank) if local_rank is not None else None,
        master_addr=os.environ.get("MASTER_ADDR"),
        master_port=int(master_port) if master_port is not None else None,
        candidate_id=None,
    )


def _worker_environment(env: Mapping[str, str] | None) -> dict[str, str]:
    process_env = os.environ.copy()
    current_pythonpath = process_env.get("PYTHONPATH")
    extra_paths = [path for path in sys.path if path]

    if current_pythonpath:
        extra_paths.append(current_pythonpath)

    process_env["PYTHONPATH"] = os.pathsep.join(extra_paths)

    if env is not None:
        process_env.update(env)

    return process_env


def _read_stderr(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
