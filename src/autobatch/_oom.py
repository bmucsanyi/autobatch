import torch

OOM_MESSAGE_PARTS = (
    "out of memory",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "cufft_alloc_failed",
    "memory allocation failed",
)


AMBIGUOUS_MESSAGE_PARTS = (
    "cudnn_status_not_supported",
    "cudnn_status_bad_param",
    "cublas_status_not_supported",
    "invalid configuration argument",
)


def is_oom_error(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True

    message = f"{type(error).__name__}: {error}".lower()

    for part in AMBIGUOUS_MESSAGE_PARTS:
        if part in message:
            return False

    return any(part in message for part in OOM_MESSAGE_PARTS)
