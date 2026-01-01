import random
from typing import Callable, Optional, List
from .block_graph import BlockGraph, BlockNode, BranchOutcome
from .code_block_generator import BlockGenerator
from .input_template import InputTemplate
from .simulation import SimulationContext

def generate_block_and_split(
    graph: BlockGraph,
    block_gen: BlockGenerator,
    template: InputTemplate,
    context: SimulationContext,
    commit: bool = True,
) -> BlockNode:
    """
    Generates a code block using `block_gen`, splits it into TAKEN/NOT-TAKEN
    paths, and adds it to the graph.

    Returns:
        The created BlockNode.
    """

    node = graph.allocate_and_add_node(template, context)

    block_gen.extend_block(node)

    if commit:
        graph.commit_node(node)

    return node


def generate_graph(
    graph: BlockGraph,
    block_gen: BlockGenerator,
    root_template: InputTemplate,
    root_context: SimulationContext,
    max_blocks: int = 10,
    prune_decider: Optional[Callable[[BlockNode, BranchOutcome], bool]] = None,
    connect_decider: Optional[Callable[[BlockNode, BranchOutcome, BlockNode], bool]] = None,
    early_stopping: Optional[Callable[BlockGraph, bool]] = None
) -> Optional[BlockNode]:
    """
    Builds a full block graph using code block generation.

    Returns the root node.
    """
    pruned_branches: List[Tuple[BlockNode, BranchOutcome]] = []
    prune_decider = prune_decider or (lambda node, outcome: False)
    connect_decider = connect_decider or (lambda src, outcome, new: False)
    early_stopping = early_stopping or (lambda graph: False)

    if early_stopping(graph):
        return None

    root = generate_block_and_split(graph, block_gen, root_template, root_context)
    frontier: List[Tuple[BlockNode, BranchOutcome]] = [
        (root, BranchOutcome.TAKEN),
        (root, BranchOutcome.NOT_TAKEN)
    ]
    random.shuffle(frontier)

    while frontier and len(graph.nodes) < max_blocks and not early_stopping(graph):
        from_node, outcome = frontier.pop(0)

        if prune_decider(from_node, outcome):
            pruned_branches.append((from_node, outcome))
            continue

        output_template = from_node.output_templates[outcome]
        output_context = from_node.output_contexts[outcome]
        new_node = generate_block_and_split(graph, block_gen, output_template, output_context)

        from_node.connect_to(outcome, new_node)

        # Connect previously pruned branches if applicable
        to_remove = []
        for idx, (pruned_src, pruned_outcome) in enumerate(pruned_branches):
            if connect_decider(pruned_src, pruned_outcome, new_node):
                pruned_src.connect_to(pruned_outcome, new_node)
                to_remove.append(idx)
        for idx in reversed(to_remove):
            pruned_branches.pop(idx)

        new_pairs = [
            (new_node, BranchOutcome.TAKEN),
            (new_node, BranchOutcome.NOT_TAKEN)
        ]
        for pair in new_pairs:
            index = random.randrange(len(frontier) + 1)
            frontier.insert(index, pair)

    # Connect all remaining pruned branches to the last dummy node
    end = graph.allocate_empty()
    for pruned_src, pruned_outcome in pruned_branches:
        pruned_src.connect_to(pruned_outcome, end)

    return root

