from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProbeStage:
    local: Callable[[int], None]
    sync: Callable[[int, bool], None]

    def __post_init__(self) -> None:
        if not callable(self.local):
            msg = "probe stage local must be callable"
            raise TypeError(msg)

        if not callable(self.sync):
            msg = "probe stage sync must be callable"
            raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class StagedProbe:
    stages: tuple[ProbeStage, ...]

    def __init__(self, stages: Sequence[ProbeStage]) -> None:
        object.__setattr__(self, "stages", tuple(stages))

        if len(self.stages) == 0:
            msg = "staged probe must contain at least one stage"
            raise ValueError(msg)

        for stage in self.stages:
            if not isinstance(stage, ProbeStage):
                msg = "staged probe stages must be ProbeStage"
                raise TypeError(msg)
