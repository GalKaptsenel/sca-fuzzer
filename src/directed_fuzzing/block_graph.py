from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Any, List
import copy
from enum import Enum

from .code_block import CodeBlock, CodeAllocator
from .input_template import InputTemplate
from .microarch_simulators import SimulationContext
from .common import BranchOutcome

@dataclass
class BlockNode:
    """
    Node in the generated-code graph.
    Holds a CodeBlock and the InputTemplate that was used as input to generate it,
    plus the resulting output InputTemplates for each branch outcome.
    """
    code_block: CodeBlock
    input_template: Optional[InputTemplate] = None                  # InputTemplate before executing the code block
    input_context: Optional[SimulationContext] = None               # SimulationContext before executing the code block

    simulation_contexts: Dict[BranchOutcome, Optional[SimulationContext]] = field( # arch + microarch context AFTER executing code block
        default_factory=lambda: {BranchOutcome.TAKEN: None, BranchOutcome.NOT_TAKEN: None}
    )

    output_templates: Dict[BranchOutcome, Optional[InputTemplate]] = field(
        default_factory=lambda: {BranchOutcome.TAKEN: None, BranchOutcome.NOT_TAKEN: None}
    )

    successors: Dict[BranchOutcome, Optional[BlockNode]] = field(
        default_factory=lambda: {BranchOutcome.TAKEN: None, BranchOutcome.NOT_TAKEN: None}
    )

    committed: bool = False

    def connect_to(self, outcome: BranchOutcome, node: BlockNode):
        branch_instruction = block_node.code_block.instructions[-1]
        assert branch_instruction.control_flow, "Last instruction in a code block is expected to be a control flow instruction"
        branch_instruction.add_op(node.code_block.label)
        self.successors[outcome] = node

    def set_output_template(self, outcome: BranchOutcome, tpl: Optional[InputTemplate]):
        self.output_templates[outcome] = tpl


@dataclass
class BlockGraph:
    """
    Graph managing BlockNodes. Supports adding nodes, searching nodes (for connecting/pruning),
    and policies to choose an existing node to attach to.
    """
    allocator: CodeAllocator
    nodes: List[BlockNode] = field(default_factory=list)

    def allocate_and_add_node(self, initial_template: InputTemplate, initial_context: SimulationContext) -> BlockNode:
        code_block = self.allocator.allocate_block()
        node = BlockNode(code_block=code_block, input_template=initial_template, input_context=initial_context)
        self.nodes.append(node)
        return node

    def commit_node(self, node: BlockNode):
        if node.committed:
            raise RuntimeError(f"Node {node} already committed")
        node.code_block.commit()
        node.committed = True

    def find_candidate_target(self, policy: Callable[[BlockNode, BranchOutcome, List[BlockNode]], Optional[BlockNode]],
                              src_node: BlockNode, outcome: BranchOutcome) -> Optional[BlockNode]:
        """
        Use policy to choose an existing node to attach to. Policy returns a node or None.
        Example policies: nearest-forward, random, match-by-template, etc.
        """
        return policy(src_node, outcome, self.nodes)

    def connect(self, src: BlockNode, outcome: BranchOutcome, dst: BlockNode):
        src.set_successor(outcome, dst)

    def allocate_empty(self):
        code_block = self.allocator.allocate_block()
        node = BlockNode(code_block=code_block)
        self.nodes.append(node)
        return node

