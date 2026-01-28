from dataclasses import dataclass
from .arch_simulator import ArchSnapshotInterface
from .microarch_state import MicroarchState


@dataclass
class SimulationContext:
    arch_snapshot: ArchSnapshotInterface
    mu_state: MicroarchState


