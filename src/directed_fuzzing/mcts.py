import math
from typing import Optional, Any, List, Tuple, Callable

from .microarch import MicroarchState
from .input_tamplate import InputTemplate
from .microarch_simulators import SimulatorInterface
from .input_generator import InputGenerator
from .scorer import ScorerInterface

from .interfaces import Instruction

@dataclass
class MCTSNode:
    parent: Optional[MCTSNode]
    instr_seq: List[Instruction] = field(default_factory=list)
    state: Optional[MicroarchState] = None
    ctx: Optional[InputTemplate] = None

    children: List[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    # actions are candidate instructions not yet expanded at this node
    untried_actions: List[Instruction] = field(default_factory=list)

    def value(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def uct_score(self, c: float) -> float:
        if 0 == self.visits:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return self.value() + c * math.sqrt(math.log(max(1, parent_visits)) / self.visits)

    def best_child(self, exploration_c) -> MCTSNode:
        return max(self.children, key=lambda child: child.uct_score(exploration_c))


class MCTS:
    def __init__(
        self,
        simulator: SimulatorInterface,
        instr_generator: Any,
        scorer: ScorerInterface,
        initial_state_factory: Callable[[], MicroarchState],
        initial_ctx_factory: Callable[[], InputTemplate],
        exploration_c: float = 1.4,
        pw_k: Optional[float] = 2.0,
        pw_alpha: float = 0.5,
        rollout_depth: int = 8,
        instructions_per_node: int = 1,
        untried_action_batch_size: int = 8,
        max_rollout_depth: int = int(1e6)
    ):
        self.simulator = simulator
        self.instr_generator = instr_generator
        self.scorer = scorer
        self.initial_state_factory = initial_state_factory
        self.initial_ctx_factory = initial_ctx_factory
        self.exploration_c = exploration_c
        self.pw_k = pw_k
        self.pw_alpha = pw_alpha
        self.rollout_depth = rollout_depth
        self.instructions_per_node = instructions_per_node
        self.untried_action_batch_size = untried_action_batch_size
        self.max_rollout_depth = max_rollout_depth

        self.root: MCTSNode = self._create_root(initial_state_factory, initial_ctx_factory)


    @staticmethod
    def _create_root(initial_state_factory, initial_ctx_factory) -> MCTSNode:
        init_state = initial_state_factory()
        init_ctx = initial_ctx_factory()
        return MCTSNode(parent=None, instr_seq=[], state=init_state, ctx=init_ctx)


    def _select(self, node: MCTSNode) -> MCTSNode:
        # Walk the tree until we find a node that can be widened (has untried actions available).
        cur = node
        while True:
            if self._can_widen(cur):
                return cur
            if not cur.children:
                return cur
            cur = cur.best_child(self.exploration_c)
        return cur

    def _can_widen(self, node: MCTSNode) -> bool:
        if self.pw_k is None:
            return bool(node.untried_actions)
        limit = int(self.pw_k * (max(1, node.visits) ** self.pw_alpha))
        return len(node.children) < limit and bool(node.untried_actions)

    def _expand(self, node: MCTSNode) -> MCTSNode:

        # Refill untried actions if empty
        if not node.untried_actions:

            for _ in range(self.untried_action_batch_size):
                candidate_batch = self.instr_generator.sample(self.instructions_per_node)
                node.untried_actions.append(candidate_batch)

            # Nothing to expand
            if not node.untried_actions:
                return node

        # pop one action (instruction batch) and simulate it
        instr_batch = node.untried_actions.pop(0)

        new_state, new_ctx = self.simulator.simulate(
                instructions=instr_batch,
                input_template=node.ctx,
                initial_state=node.state
        )

        child_seq = node.instr_seq + instr_batch
        child = MCTSNode(parent=node, instr_seq=child_seq, state=new_state, ctx=new_ctx)

        node.children.append(child)

        return child

    def _rollout(self, node: MCTSNode) -> float:
        state: MicroarchState = node.state
        ctx: InputTemplate = node.ctx

        depth_factor = max(1, len(node.instr_seq)/self.instructions_per_node)
        simulation_instruction_count = max(1, min(self.instructions_per_node * self.rollout_depth, int(self.max_rollout_depth / depth_factor)))
        instr_batch: List[Instruction] = self.instr_generator.sample(simulation_instruction_count)

        state, ctx = self.simulator.simulate(
                instructions=instr_batch,
                input_template=ctx,
                initial_state=state
            )

        reward: float = self.scorer(state)
        return reward


    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _best_sequence(self) -> Tuple[List[Instruction], InputTemplate]:
        node = self.root
        while node.children:
            node = node.best_child(self.exploration_c)
        return node.instr_seq, node.ctx


    def search(self, iterations: int = 1000) -> Tuple[List[Instruction], InputTemplate]:
        """
        Run MCTS for given iterations and return the best sequence found and its InputTemplate
        """
        for _ in range(iterations):
            node = self._select(self.root)
            node_to_run = self._expand(node)
            reward = self._rollout(node_to_run)
            self._backpropagate(node_to_run, reward)
        return self._best_sequence()

