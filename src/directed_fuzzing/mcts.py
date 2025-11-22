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
from .generation_policy import GenerationPolicy
from .common import BranchOutcome, BranchType
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
        scorer: ScorerInterface,
        initial_context_factory: Callable[[], SimulationContext],
        initial_template_factory: Callable[[], InputTemplate],
        generation_policy: GenerationPolicy,
        iterations: int = 1000,
        exploration_c: float = 1.4,
        pw_k: Optional[float] = 2.0,
        pw_alpha: float = 0.5,
        rollout_depth: int = 8,
        max_instructions_batch_size: int = 1,
        variants_per_batch: int = 8,
        max_rollout_depth: int = int(1e6),
        verbose: bool = False
    ):
        self._mu_simulator = mu_simulator
        self._instr_generator = instr_generator
        self._scorer = scorer
        self._generation_policy = generation_policy 
        self._iterations = iterations
        self._exploration_c = exploration_c
        self._pw_k = pw_k
        self._pw_alpha = pw_alpha
        self._rollout_depth = rollout_depth
        self._max_instructions_batch_size = max_instructions_batch_size
        self._variants_per_batch= variants_per_batch
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
            return bool(node.instr_seq)
        limit = int(self._pw_k * (max(1, node.visits) ** self._pw_alpha))
        return len(node.children) < limit and bool(node.untried_actions)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions:
            number_of_instructions_to_gen = min(
                    self._max_instructions_batch_size,
                    self.generation_policy.remaining_instructions(node.instr_seq)
            )

            if 0 >= number_of_instructions_to_gen:
                return node

            for _ in range(self._variants_per_batch):
                candidate_batch: List[Instruction] = self._instr_generator.sample(number_of_instructions_to_gen)
                if not candidate_batch:
                    continue
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

        depth_factor = max(1.0, len(node.instr_seq) / max(1, self._max_instructions_batch_size))
        max_possible_rollouts = int(self._max_rollout_depth / depth_factor)
        simulation_instruction_count = max(1, min(self._max_instructions_batch_size * self._rollout_depth, max_possible_rollouts))
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



    def _generate_conditional_branch(
            self,
            node: mctsnode,
            block_node: blocknode,
        ) -> list[instruction]:
        """
        generate a conditional + unconditional branch pair.
        update block_node with *two* outgoing contexts and templates.
    
        returns:
            [cond_branch, uncond_branch] or [] if impossible.
        """
        context  = node.context
        template = node.template
    
        free_regs = [r for r in template.regs if not template.is_concrete(r)]
    
        cond_branch, name, taken_val, not_taken_val = (
            self._instr_generator.generate_conditional_branch(
                free_regs=free_regs,
            )
        )
    
        if cond_branch is None:
            return []
    
        # split the template into TAKEN and NOT_TAKEN constraints
        template_taken     = template.clone().set_concrete(name, taken_val)
        template_not_taken = template.clone().set_concrete(name, not_taken_val)
    
        uncond_branch = self._instr_generator.generate_unconditional_branch()
        instr_seq = [cond_branch, uncond_branch]
    
        taken_trace, taken_context = self._mu_simulator.simulate(
            sim_context=context,
            instructions=instr_seq,
            input_template=template_taken,
        )
        final_template_taken = taken_trace.input_templates[-1]
    
        not_taken_trace, not_taken_context = self._mu_simulator.simulate(
            sim_context=context,
            instructions=instr_seq,
            input_template=template_not_taken,
        )
        final_template_not_taken = not_taken_trace.input_templates[-1]
    
        # store results
        block_node.contexts[BranchOutcome.TAKEN]     = taken_context
        block_node.contexts[BranchOutcome.NOT_TAKEN] = not_taken_context
    
        block_node.output_templates[BranchOutcome.TAKEN]     = final_template_taken
        block_node.output_templates[BranchOutcome.NOT_TAKEN] = final_template_not_taken
    
        block_node.code_block.instructions.extend(instr_seq)
    
        return instr_seq

    def _generate_unconditional_branch(
            self,
            node: MCTSNode,
            block_node: BlockNode,
        ) -> List[Instruction]:
        """
        Generate a terminal unconditional branch.
        update block_node with only TAKEN outgoing context and template.

        Returns:
            [uncond_branch]
        """

        context  = node.context
        template = node.template
    
        uncond_branch = self._instr_generator.generate_unconditional_branch()
        instr_seq = [uncond_branch]
    
        trace, new_context = self._mu_simulator.simulate(
            sim_context=context,
            instructions=instr_seq,
            input_template=template,
        )
        final_template = trace.input_templates[-1]
    
        block_node.contexts[BranchOutcome.TAKEN]         = new_context
        block_node.output_templates[BranchOutcome.TAKEN] = final_template
    
        block_node.code_block.instructions.extend(instr_seq)
    
        return instr_seq


    def _generate_unconditional_branch(
            self,
            node: MCTSNode,
            block_node: BlockNode,
        ) -> List[Instruction]:
        """
        Generate a terminal unconditional branch.
        Returns only one branch outcome.

        Returns:
            [uncond_branch]
        """
        context  = node.context
        template = node.template
        uncond_branch = self._instr_generator.generate_unconditional_branch()
        instr_seq = [uncond_branch]

        trace, new_context = self._mu_simulator.simulate(
                sim_context=context,
                instructions=instr_seq,
                input_template=template,
            )
        final_template = trace.input_templates[-1]
        
        block_node.contexts[BranchOutcome.TAKEN]          = new_context
        block_node.output_templates[BranchOutcome.TAKEN]  = final_template
        
        block_node.code_block.instructions.extend(instr_seq)

        return instr_seq

    def extend_block(
            self,
            block_node: BlockNode,
            iterations: Optional[int] = None,
        ):
        """
        Run MCTS, extract best instruction sequence,
        Append it to the block, terminate with a branch.
        """
        iterations = iterations or self._iterations
    
        for i in range(iterations):
            node = self._select(self._root)
            node_to_run = self._expand(node)
            reward = self._rollout(node_to_run)
            self._backpropagate(node_to_run, reward)
    
            if self._verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{iterations}")
    
        best_node = self._best_node()
    
        seq = best_node.instr_seq
        block_node.code_block.instructions.extend(seq)
    
        branch_type: BranchType = self._generation_policy.choose_branch_type(seq)
    
        if branch_type == BranchType.DIRECT_COND:
            success = self._generate_conditional_branch(best_node, block_node)
    
            if not success:
                self._generate_unconditional_branch(best_node, block_node)
    
        elif branch_type == BranchType.DIRECT_UNCOND:
            self._generate_unconditional_branch(best_node, block_node)
    
        else:
            raise AttributeError(f"Not-Supported BranchType: {branch_type}")
    
        if self._verbose:
            print(f"Extended block with {len(seq)} instructions (+ {branch_type.name} termination)")


