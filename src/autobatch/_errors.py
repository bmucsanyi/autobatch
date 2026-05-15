class AutobatchError(Exception):
    pass


class InvalidConfigurationError(AutobatchError):
    pass


class NoSafeValueError(AutobatchError):
    pass


class WorkloadError(AutobatchError):
    pass


class ProbeTimeoutError(AutobatchError):
    pass


class DistributedError(AutobatchError):
    pass
