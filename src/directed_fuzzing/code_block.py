from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, List
from ..interfaces import Instruction
from .common import BranchOutcome


class CodeBlock:
    def __init__(self, code_allocator: CodeAllocator):
        self.instructions: List[Instruction] = []

        self.successors: Dict[BranchOutcome, Optional[CodeBlock]] = {
            BranchOutcome.TAKEN: None,
            BranchOutcome.NOT_TAKEN: None
        }

        self._allocator = code_allocator

    @property
    def start_address(self) -> int:
        return self._allocator.get_block_address(self)

    @property
    def size(self) -> int:
        # AArch64 has 4-byte fixed width instructions
        return len(self.instructions) * 4

    def commit(self):
        self._allocator.commit_block(self)

    def __repr__(self):
        return f"<CodeBlock at 0x{self.start_address:x}, size={self.size} bytes>"


class CodeAllocator:
    def __init__(self, base_address: int):
        # Ensure 4-byte alignment for AArch64 instructions
        if base_address % 4 != 0:
            raise ValueError(f"Base address must be 4-byte aligned, got 0x{base_address:x}")

        self._base_address = base_address
        self._write_ptr = base_address

        # Mapping of blocks
        self._block_to_address: Dict[CodeBlock, int] = {}
        self._block_to_committed_flag: Dict[CodeBlock, bool] = {}

    def allocate_block(self) -> CodeBlock:
        block = CodeBlock(self)
        self._block_to_address[block] = self._write_ptr
        self._block_to_committed_flag[block] = False
        return block

    def get_block_address(self, block: CodeBlock) -> int:
        if not isinstance(block, CodeBlock):
            raise ValueError(f"Unexpected block type: {type(block)}")

        if block not in self._block_to_address:
            raise ValueError(f"Unknown block: {block}")

        return self._block_to_address[block]

    def commit_block(self, block: CodeBlock):
        if not isinstance(block, CodeBlock):
            raise ValueError(f"Unexpected block type to be committed: {type(block)}")

        if block not in self._block_to_committed_flag:
            raise ValueError(f"Unknown block: {block}")

        if self._block_to_committed_flag[block]:
            raise RuntimeError(f"Block already committed at 0x{self.get_block_address(block):x}")

        # Enforce sequential writing of blocks
        allocated_addr = self.get_block_address(block)
        if allocated_addr != self._write_ptr:
            raise RuntimeError(
                f"Out-of-order block commit! "
                f"Block allocated at 0x{allocated_addr:x}, "
                f"write_ptr at 0x{self._write_ptr:x}"
            )

        # Increment write pointer by block size
        self._write_ptr += block.size
        self._block_to_committed_flag[block] = True

        assert self._write_ptr & 3 == 0, f"self._write_ptr expected to always be 4-bytes aligned, got: 0x{self._write_ptr:x}"

    def __repr__(self):
        return f"<CodeAllocator base=0x{self._base_address:x}, write_ptr=0x{self._write_ptr:x}>"

