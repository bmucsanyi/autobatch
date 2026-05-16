class AutobatchError(Exception):
    pass


class InvalidConfigurationError(AutobatchError):
    pass


class NoSafeValueError(AutobatchError):
    pass


class ProbeError(AutobatchError):
    pass


class DistributedError(AutobatchError):
    pass
