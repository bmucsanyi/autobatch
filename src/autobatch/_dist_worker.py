import os

import torch.distributed as dist

from autobatch._errors import DistributedError
from autobatch._protocol import ProbeRequest


def setup_candidate_process_group(request: ProbeRequest) -> bool:
    if os.environ.get("AUTOBATCH_WORKER_DISTRIBUTED") != "1":
        return False

    if request.rank is None or request.world_size is None:
        msg = "distributed worker request is missing rank geometry"
        raise DistributedError(msg)

    if request.master_addr is None or request.master_port is None:
        msg = "distributed worker request is missing rendezvous"
        raise DistributedError(msg)

    if dist.is_initialized():
        return False

    os.environ["MASTER_ADDR"] = request.master_addr
    os.environ["MASTER_PORT"] = str(request.master_port)
    os.environ["RANK"] = str(request.rank)
    os.environ["WORLD_SIZE"] = str(request.world_size)

    if request.local_rank is not None:
        os.environ["LOCAL_RANK"] = str(request.local_rank)

    dist.init_process_group(backend="nccl")

    return True


def teardown_candidate_process_group(created: bool) -> None:
    if not created:
        return

    if dist.is_initialized():
        dist.destroy_process_group()
