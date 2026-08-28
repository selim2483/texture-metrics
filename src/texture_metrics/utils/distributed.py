import os

import idr_torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_local_rank() -> int:
    if os.environ.get("CLUSTER_NAME") == "telecom":
        return int(os.environ.get("LOCAL_RANK", 0))
    else:
        return idr_torch.local_rank


def get_world_size() -> int:
    if os.environ.get("CLUSTER_NAME") == "telecom":
        return int(os.environ.get("WORLD_SIZE", 1))
    else:
        return idr_torch.world_size


def get_global_rank() -> int:
    if os.environ.get("CLUSTER_NAME") == "telecom":
        return int(os.environ.get("RANK", 0))
    else:
        return idr_torch.rank
