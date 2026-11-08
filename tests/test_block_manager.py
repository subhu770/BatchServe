"""Tests for the Paged KV-Cache Block Allocator."""

import pytest
from app.models.sequence import Sequence
from app.services.block_manager import BlockManager
from tests.conftest import create_mock_sequence


def test_pool_initialization(block_manager: BlockManager) -> None:
    """Verifies that the block pool initializes with expected block counts."""
    assert block_manager.total_blocks == 16
    assert block_manager.num_free_blocks == 16
    assert block_manager.num_used_blocks == 0
    assert block_manager.kv_cache_utilization == 0.0


def test_prompt_block_allocation(block_manager: BlockManager) -> None:
    """Verifies block count math for prompt token allocation."""
    # block_size = 4, prompt length = 7 -> ceil(7/4) = 2 blocks
    seq = create_mock_sequence(seq_id="seq-1", prompt_tokens=[10, 11, 12, 13, 14, 15, 16])

    assert block_manager.can_allocate(seq) is True
    blocks = block_manager.allocate(seq)

    assert len(blocks) == 2
    assert block_manager.num_used_blocks == 2
    assert block_manager.num_free_blocks == 14
    assert block_manager.get_block_table(seq.seq_id) == blocks


def test_append_token_boundary_math(block_manager: BlockManager) -> None:
    """Verifies dynamic allocation when token step crosses block boundary."""
    # block_size = 4, prompt has 4 tokens -> fits in 1 block exactly
    seq = create_mock_sequence(seq_id="seq-2", prompt_tokens=[1, 2, 3, 4])
    initial_blocks = block_manager.allocate(seq)
    assert len(initial_blocks) == 1

    # seq.total_tokens is 4. Generating 5th token crosses boundary (4 % 4 == 0)
    assert block_manager.can_append_token(seq) is True
    new_block = block_manager.append_token(seq)
    assert new_block is not None
    assert len(block_manager.get_block_table(seq.seq_id)) == 2

    # Simulate token addition
    seq.output_tokens.append(100)  # total_tokens is now 5
    # For tokens 5, 6, 7 (total_tokens % 4 != 0), fits in existing block
    assert block_manager.append_token(seq) is None
    seq.output_tokens.append(101)  # total 6
    assert block_manager.append_token(seq) is None
    seq.output_tokens.append(102)  # total 7
    assert block_manager.append_token(seq) is None

    # Now total_tokens is 8 (8 % 4 == 0) -> next token needs a 3rd block
    seq.output_tokens.append(103)  # total 8
    third_block = block_manager.append_token(seq)
    assert third_block is not None
    assert len(block_manager.get_block_table(seq.seq_id)) == 3


def test_zero_kv_cache_leaks_on_deallocation(block_manager: BlockManager) -> None:
    """Guarantees zero memory leaks after sequence completion and deallocation."""
    seq1 = create_mock_sequence(seq_id="seq-leak-1", prompt_tokens=[1, 2, 3, 4, 5, 6, 7])
    seq2 = create_mock_sequence(seq_id="seq-leak-2", prompt_tokens=[8, 9, 10, 11])

    block_manager.allocate(seq1)
    block_manager.allocate(seq2)
    assert block_manager.num_used_blocks == 3  # 2 + 1

    # Append a token to seq2
    block_manager.append_token(seq2)  # 4 tokens -> crosses boundary to block 2
    assert block_manager.num_used_blocks == 4

    # Deallocate both sequences
    block_manager.free(seq1)
    assert block_manager.num_used_blocks == 2
    assert block_manager.get_block_table(seq1.seq_id) == []

    block_manager.free(seq2)
    assert block_manager.num_used_blocks == 0
    assert block_manager.num_free_blocks == block_manager.total_blocks
    assert block_manager.kv_cache_utilization == 0.0
    assert len(block_manager.block_tables) == 0


def test_out_of_memory_handling() -> None:
    """Tests MemoryError when physical pool is exhausted."""
    small_manager = BlockManager(total_blocks=2, block_size=4, enable_prefix_caching=False)
    seq1 = create_mock_sequence(seq_id="seq-oom-1", prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8])  # 2 blocks
    seq2 = create_mock_sequence(seq_id="seq-oom-2", prompt_tokens=[9, 10, 11, 12])  # 1 block

    small_manager.allocate(seq1)
    assert small_manager.num_free_blocks == 0
    assert small_manager.can_allocate(seq2) is False

    with pytest.raises(MemoryError, match="Out of KV-cache memory"):
        small_manager.allocate(seq2)


def test_prefix_caching_sharing(prefix_block_manager: BlockManager) -> None:
    """Tests that identical prompt block prefixes share physical blocks."""
    # Both sequences share prompt prefix [10, 20, 30, 40]
    seq1 = create_mock_sequence(seq_id="seq-pfx-1", prompt_tokens=[10, 20, 30, 40, 50])
    seq2 = create_mock_sequence(seq_id="seq-pfx-2", prompt_tokens=[10, 20, 30, 40, 60])

    blocks1 = prefix_block_manager.allocate(seq1)
    assert len(blocks1) == 2  # 1 prefix block + 1 unaligned block
    assert prefix_block_manager.num_used_blocks == 2

    # Second sequence should hit prefix cache for the first block
    blocks2 = prefix_block_manager.allocate(seq2)
    assert blocks1[0] == blocks2[0]  # Shared physical block!
    assert blocks1[1] != blocks2[1]  # Independent partial blocks
    assert prefix_block_manager.physical_blocks[blocks1[0]].ref_count == 2
    assert prefix_block_manager.cache_hits >= 1

    # Free sequence 1: shared block should stay active (ref_count decremented to 1)
    prefix_block_manager.free(seq1)
    assert prefix_block_manager.physical_blocks[blocks1[0]].ref_count == 1
    assert prefix_block_manager.num_used_blocks == 2  # shared block + seq2's partial block

    # Free sequence 2: shared block ref_count becomes 0, returned to free pool
    prefix_block_manager.free(seq2)
    assert prefix_block_manager.num_free_blocks == prefix_block_manager.total_blocks
    assert prefix_block_manager.num_used_blocks == 0

