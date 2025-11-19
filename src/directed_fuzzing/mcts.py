from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Any, List, Tuple, Callable, Deque

import math

from .input_template import InputTemplate
from .microarch_simulators import MicroarchSimulatorInterface, SimulationContext
from .scorer import ScorerInterface
from .code_block import CodeBlock
from .code_block_generator import CodeBlockGenerator 
from .common import BranchOutcome
from ..interfaces import Instruction

@dataclass
class MCTSNode:
    parent: Optional[MCTSNode]
    context: SimulationContext
    template: InputTemplate
    instr_seq: List[Instruction] = field(default_factory=list)
    children: List[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    # untried actions are candidate instruction (batches) not yet expanded at this node
    untried_actions: Deque[List[Instruction]] = field(default_factory=deque)

    def value(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def uct_score(self, c: float) -> float:
        if 0 == self.visits:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return self.value() + c * math.sqrt(math.log(max(1, parent_visits)) / self.visits)

    def best_child(self, exploration_c) -> MCTSNode:
        return max(self.children, key=lambda child: child.uct_score(exploration_c))


class MCTS(CodeBlockGenerator):
    def __init__(
        self,
        mu_simulator: MicroarchSimulatorInterface,
        instr_generator: Any,
        max_instructions: int = 8,
        scorer: ScorerInterface,
        initial_context_factory: Callable[[], SimulationContext],
        initial_template_factory: Callable[[], InputTemplate],
        iterations: int = 1000,
        exploration_c: float = 1.4,
        pw_k: Optional[float] = 2.0,
        pw_alpha: float = 0.5,
        rollout_depth: int = 8,
        max_instructions_per_node: int = 1,
        untried_action_batch_size: int = 8,
        max_rollout_depth: int = int(1e6),
        verbose: bool = False
    ):
        self._mu_simulator = mu_simulator
        self._instr_generator = instr_generator
        self._scorer = scorer
        self._max_instructions = max_instruction
        self._iterations = iterations
        self._exploration_c = exploration_c
        self._pw_k = pw_k
        self._pw_alpha = pw_alpha
        self._rollout_depth = rollout_depth
        self._max_instructions_per_node = max_instructions_per_node
        self._untried_action_batch_size = untried_action_batch_size
        self._max_rollout_depth = max_rollout_depth
        self._verbose = verbose

        self._root: MCTSNode = self._create_root(initial_context_factory, initial_template_factory)


    @staticmethod
    def _create_root(initial_context_factory, initial_template_factory) -> MCTSNode:
        init_context = initial_context_factory()
        init_template = initial_template_factory()
        return MCTSNode(parent=None, context=init_context, template=init_template)

    def _select(self, node: MCTSNode) -> MCTSNode:
        cur = node
        while True:
            if self._can_widen(cur):
                return cur
            if not cur.children:
                return cur
            cur = cur.best_child(self._exploration_c)

    def _can_widen(self, node: MCTSNode) -> bool:
        if self._pw_k is None:
            return bool(node.untried_actions)
        limit = int(self._pw_k * (max(1, node.visits) ** self._pw_alpha))
        return len(node.children) < limit and bool(node.untried_actions)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions:
            for _ in range(self._untried_action_batch_size):
                number_of_instructions_to_gen = min(
                        self._max_instructions_per_node,
                        self._max_instructions - len(node.instr_seq)
                )
                candidate_batch = self._instr_generator.sample(number_of_instructions_to_gen)
                node.untried_actions.append(candidate_batch)

            # Nothing to expand
            if not node.untried_actions:
                return node

        # pop one action (instruction batch) and simulate it
        instr_batch = node.untried_actions.popleft()

        trace, new_context = self._mu_simulator.simulate(
                sim_context=node.context,
                instructions=instr_batch,
                input_template=node.template,
        )

        assert trace.input_templates
        assert trace.input_templates[0] is not node.template
        assert new_context is not node.context

        child_seq = node.instr_seq + instr_batch
        child = MCTSNode(
                parent=node,
                context=new_context,
                template=trace.input_templates[-1],
                instr_seq=child_seq
        )

        node.children.append(child)

        if self._verbose:
            print(f"Expanded node with {len(instr_batch)} instructions. Total children: {len(node.children)}")

        return child

    def _rollout(self, node: MCTSNode) -> float:
        context: SimulationContext = node.context
        template: InputTemplate = node.template

        depth_factor = max(1.0, len(node.instr_seq) / max(1, self._max_instructions_per_node))
        max_possible_rollouts = int(self._max_rollout_depth / depth_factor)
        simulation_instruction_count = max(1, min(self._max_instructions_per_node * self._rollout_depth, max_possible_rollouts))
        instr_batch: List[Instruction] = self._instr_generator.sample(simulation_instruction_count)

        _, new_context = self._mu_simulator.simulate(
                sim_context=context,
                instructions=instr_batch,
                input_template=template,
            )

        reward: float = self._scorer(new_context.mu_state)

        if self._verbose:
            print(f"Rollout reward: {reward:.4f}")

        return reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _best_node(self) -> MCTSNode:
        node = self._root

        while node.children:
            node = node.best_child(self._exploration_c)

        return node

    def extend_block(
            self,
            code_block: CodeBlock,
            iterations: Optional[int] = None
        ) -> Tuple[InputTemplate, InputTemplate]:
        """
        Run MCTS return the best sequence found and its InputTemplate
        Returns:
            - Taken InputTemplate, which represents the InputTemplate for the taken direction
            - Not Taken InputTemplate, which represents the InputTemplate for the not taken direction
        """
        iterations = iterations or self._iterations
    
        for i in range(iterations):
            node = self._select(self._root)
            node_to_run = self._expand(node)
            reward = self._rollout(node_to_run)
            self._backpropagate(node_to_run, reward)
    
            if self._verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{iterations}")
    
        best_node = self._best_node()
    
        seq = best_node.instr_seq
        code_block.instructions.extend(seq)
        code_block.simulation_context = best_node.context

        termination_instruction = self._terminate_hook(code_block)
        if branch_type == "conditional_then_unconditional":
            template_taken, template_not_taken = append_unconditional_branch(code_block)
        else:
            branch_type = "unconditional"
            template_taken, template_not_taken = append_conditional_then_unconditional_branch(code_block)
    
        if self._verbose:
            print(f"Extended block with {len(seq)} instructions (+ {branch_type} termination)")

        return template_taken, template_not_taken


