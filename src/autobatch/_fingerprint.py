import hashlib
import importlib.util
import json
import os
import platform
from collections.abc import Mapping
from pathlib import Path

import torch

from autobatch._errors import InvalidConfigurationError
from autobatch._workload import split_import_path


def canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError as error:
        msg = "cache identity values must be JSON-serializable"
        raise InvalidConfigurationError(msg) from error


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identity_key(identity: Mapping[str, object]) -> str:
    return stable_hash(identity)


def workload_source_fingerprint(workload: str) -> dict[str, object]:
    module_name, attribute_name = split_import_path(workload)
    spec = importlib.util.find_spec(module_name)

    if spec is None:
        msg = f"workload module cannot be found: {module_name}"
        raise InvalidConfigurationError(msg)

    origin = spec.origin

    if origin is None or origin == "built-in":
        msg = f"workload module has no source file: {module_name}"
        raise InvalidConfigurationError(msg)

    path = Path(origin)

    if not path.is_file():
        msg = f"workload module source cannot be read: {module_name}"
        raise InvalidConfigurationError(msg)

    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "module": module_name,
        "attribute": attribute_name,
        "origin": origin,
        "source_sha256": source_hash,
    }


def runtime_fingerprint() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    devices = []

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": properties.name,
                "total_memory": properties.total_memory,
                "major": properties.major,
                "minor": properties.minor,
            })

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": devices,
    }


def environment_fingerprint() -> dict[str, object]:
    names = (
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_WORLD_SIZE",
        "NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_BLOCKING_WAIT",
        "PYTORCH_CUDA_ALLOC_CONF",
        "WORLD_SIZE",
    )
    data = {}

    for name in names:
        if name in os.environ:
            data[name] = os.environ[name]

    return data


def distributed_fingerprint() -> dict[str, object]:
    return {
        "enabled": os.environ.get("WORLD_SIZE") is not None,
        "world_size": os.environ.get("WORLD_SIZE"),
        "local_world_size": os.environ.get("LOCAL_WORLD_SIZE"),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
