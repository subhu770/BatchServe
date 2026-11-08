"""Paged KV-Cache Memory Block Allocator.

Inspired by vLLM's PagedAttention virtual memory manager. Divides token KV-cache
into fixed-size physical memory blocks to eliminate external fragmentation
and support dynamic iteration-level continuous batching.
"""

import collections
import math
import time
from typing import Dict, List, Optional, Set, Tuple
from app.core.config import settings
from app.core.logger import logger
from app.models.sequence import Sequence


class PhysicalBlock:
    """Represents a discrete contiguous physical memory chunk in the KV-cache."""

    def __init__(self, block_id: int) -> None:
        self.block_id: int = block_id
        self.ref_count: int = 0
        self.last_accessed: float = time.time()
        self.cached_prefix_tokens: Optional[Tuple[int, ...]] = None

    def reset(self) -> None:
        """Resets physical block metadata upon deallocation."""
        self.ref_count = 0
        self.last_accessed = time.time()
        self.cached_prefix_tokens = None


class BlockManager:
    """Manages the pool of physical KV-cache blocks and sequence block tables.

    Provides exact block allocation, logical-to-physical address translation,
    prefix caching lookup, and deterministic deallocation to guarantee zero memory leaks.
    """

    def __init__(
        self,
        total_blocks: Optional[int] = None,
        total_swap_blocks: Optional[int] = None,
        block_size: Optional[int] = None,
        enable_prefix_caching: Optional[bool] = None,
    ) -> None:
        self.total_blocks: int = total_blocks or settings.TOTAL_BLOCKS
        self.total_swap_blocks: int = (
            total_swap_blocks if total_swap_blocks is not None else settings.TOTAL_SWAP_BLOCKS
        )
        self.block_size: int = block_size or settings.BLOCK_SIZE
        self.enable_prefix_caching: bool = (
            enable_prefix_caching
            if enable_prefix_caching is not None
            else settings.ENABLE_PREFIX_CACHING
        )

        # Primary Physical block storage (Active GPU/VRAM simulator)
        self.physical_blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i) for i in range(self.total_blocks)
        }
        self.free_blocks: collections.deque[int] = collections.deque(range(self.total_blocks))

        # Secondary Swap block storage (CPU RAM simulator)
        self.swap_blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i) for i in range(self.total_swap_blocks)
        }
        self.free_swap_blocks: collections.deque[int] = collections.deque(range(self.total_swap_blocks))

        # Logical-to-physical block tables
        self.block_tables: Dict[str, List[int]] = {}
        self.swap_tables: Dict[str, List[int]] = {}

        # Prefix cache mapping: token_tuple -> physical_block_id
        self.prefix_cache: Dict[Tuple[int, ...], int] = {}

        # Telemetry counters
        self.total_allocations: int = 0
        self.total_deallocations: int = 0
        self.total_swap_outs: int = 0
        self.total_swap_ins: int = 0
        self.cache_lookups: int = 0
        self.cache_hits: int = 0

    @property
    def num_free_blocks(self) -> int:
        """Returns the number of free physical blocks available."""
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        """Returns the number of physical blocks currently allocated."""
        return self.total_blocks - len(self.free_blocks)

    @property
    def kv_cache_utilization(self) -> float:
        """Calculates current KV-cache memory utilization fraction [0.0 - 1.0]."""
        if self.total_blocks == 0:
            return 0.0
        return self.num_used_blocks / self.total_blocks

    @property
    def cache_hit_rate(self) -> float:
        """Calculates cumulative prefix cache hit rate [0.0 - 1.0]."""
        if self.cache_lookups == 0:
            return 0.0
        return self.cache_hits / self.cache_lookups

    def get_block_table(self, seq_id: str) -> List[int]:
        """Returns a copy of the physical block IDs assigned to a sequence."""
        return list(self.block_tables.get(seq_id, []))

    def can_allocate(self, seq: Sequence) -> bool:
        """Evaluates whether sufficient physical blocks exist to admit the sequence."""
        prompt_len = seq.num_prompt_tokens
        if prompt_len == 0:
            return True

        blocks_needed = math.ceil(prompt_len / self.block_size)

        if not self.enable_prefix_caching:
            return len(self.free_blocks) >= blocks_needed

        # Calculate net blocks needed considering prefix cache hits
        net_needed = 0
        num_full_blocks = prompt_len // self.block_size
        for i in range(num_full_blocks):
            prefix_slice = tuple(seq.prompt_tokens[i * self.block_size : (i + 1) * self.block_size])
            if prefix_slice not in self.prefix_cache:
                net_needed += 1

        # Plus remaining partial block if prompt length not aligned
        if prompt_len % self.block_size != 0:
            net_needed += 1

        return len(self.free_blocks) >= net_needed

    def allocate(self, seq: Sequence) -> List[int]:
        """Allocates physical blocks for prompt tokens with optional prefix caching.

        Raises:
            MemoryError: If available free blocks are insufficient.
        """
        if seq.seq_id in self.block_tables:
            logger.warning("Sequence %s already has an active block table. Reallocating.", seq.seq_id)
            self.free(seq)

        prompt_len = seq.num_prompt_tokens
        blocks_needed = math.ceil(prompt_len / self.block_size) if prompt_len > 0 else 1
        allocated_block_ids: List[int] = []

        if not self.can_allocate(seq):
            raise MemoryError(
                f"Out of KV-cache memory: requested {blocks_needed} blocks, "
                f"but only {len(self.free_blocks)} available."
            )

        num_full_blocks = prompt_len // self.block_size
        for i in range(num_full_blocks):
            prefix_slice = tuple(seq.prompt_tokens[i * self.block_size : (i + 1) * self.block_size])

            if self.enable_prefix_caching:
                self.cache_lookups += 1
                if prefix_slice in self.prefix_cache:
                    cached_block_id = self.prefix_cache[prefix_slice]
                    block = self.physical_blocks[cached_block_id]
                    block.ref_count += 1
                    block.last_accessed = time.time()
                    allocated_block_ids.append(cached_block_id)
                    self.cache_hits += 1
                    continue

            # Cache miss or prefix caching disabled: pop from free pool
            block_id = self.free_blocks.popleft()
            block = self.physical_blocks[block_id]
            block.ref_count = 1
            block.last_accessed = time.time()

            if self.enable_prefix_caching:
                block.cached_prefix_tokens = prefix_slice
                self.prefix_cache[prefix_slice] = block_id

            allocated_block_ids.append(block_id)
            self.total_allocations += 1

        # Allocate partial block for remaining unaligned tokens or empty prompt
        if prompt_len % self.block_size != 0 or prompt_len == 0:
            block_id = self.free_blocks.popleft()
            block = self.physical_blocks[block_id]
            block.ref_count = 1
            block.last_accessed = time.time()
            allocated_block_ids.append(block_id)
            self.total_allocations += 1

        self.block_tables[seq.seq_id] = allocated_block_ids
        return allocated_block_ids

    def will_need_new_block(self, seq: Sequence) -> bool:
        """Determines whether the sequence will require allocating an additional block on the next step."""
        return seq.total_tokens > 0 and seq.total_tokens % self.block_size == 0

    def can_append_token(self, seq: Sequence) -> bool:
        """Checks if a new physical block is needed and available for the next token step.

        A new block is required whenever the current total sequence length is a multiple
        of block_size, transitioning into the next logical block.
        """
        if self.will_need_new_block(seq):
            return len(self.free_blocks) > 0

        # Fits in the existing physical block
        return True

    def append_token(self, seq: Sequence) -> Optional[int]:
        """Dynamically allocates an additional physical block if entering a new block boundary.

        Returns:
            The allocated block_id if a new block was added, or None if the token
            fits within the current block.

        Raises:
            MemoryError: If a new block is required but none are free.
        """
        curr_tokens = seq.total_tokens

        # Check if we crossed the block boundary
        if curr_tokens > 0 and curr_tokens % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError(
                    f"Out of KV-cache memory during generation for seq {seq.seq_id}: "
                    f"free pool is exhausted."
                )

            block_id = self.free_blocks.popleft()
            block = self.physical_blocks[block_id]
            block.ref_count = 1
            block.last_accessed = time.time()

            self.block_tables[seq.seq_id].append(block_id)
            self.total_allocations += 1
            return block_id

        # Token fits within the sequence's existing block
        return None

    def free(self, seq: Sequence) -> None:
        """Deallocates all physical blocks assigned to a sequence and returns them to free pool.

        Guarantees zero memory leaks by updating reference counts and cleaning block tables.
        """
        allocated_blocks = self.block_tables.pop(seq.seq_id, [])
        if not allocated_blocks:
            return

        for block_id in allocated_blocks:
            block = self.physical_blocks[block_id]
            block.ref_count -= 1

            if block.ref_count <= 0:
                # Remove from prefix cache if this block was cached
                if block.cached_prefix_tokens and block.cached_prefix_tokens in self.prefix_cache:
                    del self.prefix_cache[block.cached_prefix_tokens]

                block.reset()
                self.free_blocks.append(block_id)
                self.total_deallocations += 1

        logger.debug(
            "Deallocated %d blocks for seq %s. Free blocks: %d/%d",
            len(allocated_blocks),
            seq.seq_id,
            len(self.free_blocks),
            self.total_blocks,
        )

    def reset_all(self) -> None:
        """Resets the entire allocator to initial clean state."""
        self.block_tables.clear()
        self.prefix_cache.clear()
        self.free_blocks = collections.deque(range(self.total_blocks))
        for block in self.physical_blocks.values():
            block.reset()
        self.total_allocations = 0
        self.total_deallocations = 0
        self.cache_lookups = 0
        self.cache_hits = 0
