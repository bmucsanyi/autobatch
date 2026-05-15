from autobatch._oom import is_oom_error


def test_oom_classifier_accepts_clear_cuda_oom_message() -> None:
    error = RuntimeError("CUDA out of memory. Tried to allocate 20 MiB")

    assert is_oom_error(error)


def test_oom_classifier_accepts_cublas_allocation_failure() -> None:
    error = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED while creating handle")

    assert is_oom_error(error)


def test_oom_classifier_rejects_ambiguous_cudnn_message() -> None:
    error = RuntimeError("cuDNN error: CUDNN_STATUS_NOT_SUPPORTED")

    assert not is_oom_error(error)


def test_oom_classifier_rejects_illegal_memory_access() -> None:
    error = RuntimeError("CUDA illegal memory access")

    assert not is_oom_error(error)
