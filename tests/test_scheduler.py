"""Tests for Iteration-Level Continuous Batch Scheduler."""

import time
import pytest
from app.models.sequence import SequenceStatus
from app.services.block_manager import BlockManager
from app.services.scheduler import Scheduler
from tests.conftest import create_mock_sequence


def test_scheduler_admission_and_lifecycle(scheduler: Scheduler) -> None:
    """Verifies sequence admission from WAITING to RUNNING."""
    seq1 = create_mock_sequence(seq_id="s1", prompt_tokens=[1, 2, 3])
    seq2 = create_mock_sequence(seq_id="s2", prompt_tokens=[4, 5, 6])

    scheduler.add_sequence(seq1)
    scheduler.add_sequence(seq2)
    assert scheduler.num_waiting == 2
    assert scheduler.num_running == 0

    output = scheduler.schedule()
    assert len(output.running_sequences) == 2
    assert scheduler.num_waiting == 0
    assert scheduler.num_running == 2
    assert seq1.status == SequenceStatus.RUNNING
    assert seq2.status == SequenceStatus.RUNNING


def test_continuous_batching_eviction(scheduler: Scheduler) -> None:
    """Verifies dynamic eviction of finished sequences while long sequences continue."""
    seq_short = create_mock_sequence(seq_id="short", prompt_tokens=[1, 2], max_new_tokens=2)
    seq_long = create_mock_sequence(seq_id="long", prompt_tokens=[3, 4], max_new_tokens=10)

    scheduler.add_sequence(seq_short)
    scheduler.add_sequence(seq_long)
    scheduler.schedule()

    # Simulate short sequence completing
    seq_short.mark_finished("length", timestamp=time.time())

    # Next iteration step
    output = scheduler.schedule()
    assert len(output.running_sequences) == 1
    assert output.running_sequences[0].seq_id == "long"
    assert len(output.finished_sequences) == 1
    assert output.finished_sequences[0].seq_id == "short"

    # Verify short sequence blocks were freed
    assert scheduler.block_manager.get_block_table("short") == []


def test_max_batch_size_throttling(block_manager: BlockManager) -> None:
    """Verifies that running batch size does not exceed max_batch_size."""
    sched = Scheduler(block_manager=block_manager, max_batch_size=2, watermark_blocks=1)

    seqs = [create_mock_sequence(seq_id=f"seq-{i}", prompt_tokens=[i, i + 1]) for i in range(5)]
    for s in seqs:
        sched.add_sequence(s)

    out = sched.schedule()
    assert len(out.running_sequences) == 2
    assert sched.num_waiting == 3


def test_preemption_under_kv_memory_pressure() -> None:
    """Verifies preemption: victim sequence evicted to WAITING and blocks freed."""
    # Pool with 4 blocks total, block_size = 4, watermark = 1 block
    bm = BlockManager(total_blocks=4, block_size=4, enable_prefix_caching=False)
    sched = Scheduler(block_manager=bm, max_batch_size=4, watermark_blocks=1)

    # seq1 and seq2 each take 1 block initially (2 used, 2 free)
    seq1 = create_mock_sequence(seq_id="victim-1", prompt_tokens=[1, 2, 3, 4])
    seq2 = create_mock_sequence(seq_id="victim-2", prompt_tokens=[5, 6, 7, 8])

    sched.add_sequence(seq1)
    sched.add_sequence(seq2)
    sched.schedule()
    assert len(sched.running_sequences) == 2
    assert bm.num_used_blocks == 2

    # Fill both sequences to 4 total tokens so both need a new block next step
    # 2 free blocks - 2 needed = 0 free < watermark (1) -> preemption triggers!
    out = sched.schedule()

    # One sequence must be preempted to satisfy watermark headroom
    assert len(out.preempted_sequences) >= 1
    preempted_seq = out.preempted_sequences[0]
    assert preempted_seq.status == SequenceStatus.PREEMPTED
    assert bm.get_block_table(preempted_seq.seq_id) == []  # Blocks deallocated
    assert sched.num_waiting >= 1  # Re-enqueued in waiting queue

