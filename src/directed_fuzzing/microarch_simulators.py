from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass

from ..interfaces import Instruction
from .input_template import InputTemplate
from .microarch import MicroarchEvent, BPMicroarchEvent, MicroarchState
from .arch_simulator import ArchSimulatorInterface, ArchSimStep, ArchSnapshotInterface


@dataclass
class SimulationContext:
    arch_snapshot: ArchSnapshotInterface
    mu_state: MicroarchState


@dataclass
class MicroarchSimulationTrace:
    states: List[MicroarchState]             # full microarchitectural state after each instr
    templates: List[InputTemplate]     # concrete/non-concrete input template evolution
    events: List[List[MicroarchEvent]]       # microarchitectural events per instr


class MicroarchSimulatorInterface(ABC):

    @abstractmethod
    def simulate(
        self,
        sim_context: SimulationContext,
        instructions: List[Instruction],
        input_template: InputTemplate,
    ) -> MicroarchSimulationTrace:
        """Run a full microarchitectural simulation."""
        pass


class MuSimulator(MicroarchSimulatorInterface):

    def __init__(self, arch_simulator: ArchSimulatorInterface):
        if not isinstance(arch_simulator, ArchSimulatorInterface):
            raise TypeError("arch_simulator must implement ArchSimulatorInterface")
        self._arch_sim = arch_simulator


    def _step(
            self,
            instr: Instruction,
            mu_state: MicroarchState,
            input_template: InputTemplate
    ) -> Tuple[MicroarchState, InputTemplate, List[MicroarchEvent]]:

        # Clone must be deep - for state not to leak to other steps
        new_mu_state = mu_state.clone()

        step: ArchSimStep = self._arch_sim.execute_instruction(instr, input_template)
        if not isinstance(step, ArchSimStep):
            raise TypeError("arch_simulator.execute_instruction must return ArchSimStep")

        events: List[MicroarchEvent] = []

        if instr.control_flow:
            if step.taken is None:
                raise RuntimeError("Control-flow instruction must return 'taken' flag")

            prediction = mu_state.bp.predict(step.instruction_address)

            events.append(
                BPMicroarchEvent(
                    pc=step.instruction_address,
                    taken=step.taken,
                    prediction=prediction
                )
            )

            new_mu_state.bp.update(step.instruction_address, step.taken)

        return new_mu_state, step.updated_input_template, events

    def simulate(
            self,
            sim_context: SimulationContext,
            instructions: List[Instruction],
            input_template: InputTemplate,
        ) -> Tuple[MicroarchSimulationTrace, SimulationContext]:

        mu_state = sim_context.mu_state.clone()
        template = input_template.clone()
        states: List[MicroarchState] = [mu_state]
        templates: List[InputTemplate] = [template]
        events_per_instr: List[List[MicroarchEvent]] = [[]]

        self._arch_sim.restore_snapshot(sim_context.arch_snapshot)

        for instr in instructions:
            mu_state, template, ev = self._step(instr, mu_state, template)

            states.append(mu_state)
            templates.append(template)
            events_per_instr.append(ev)

        out_context = SimulationContext(
                arch_snapshot=self._arch_sim.take_snapshot(),
                mu_state=states[-1]
        )

        trace = MicroarchSimulationTrace(
            states=states,
            templates=templates,
            events=events_per_instr
        )

        return trace, out_context
