"""Iteration-Level Continuous Batch Scheduler.

Dynamically manages inference sequence lifecycles across WAITING, RUNNING,
PREEMPTED, and FINISHED states. Performs iteration-step continuous batching
with memory-aware admission control and preemption.
"""

import collections
import time
from dataclasses import dataclass, field
from typing import Deque, List, Optional
from app.core.config import settings
from app.core.logger import logger
from app.models.sequence import Sequence, SequenceStatus
from app.services.block_manager import BlockManager


@dataclass
class SchedulerOutput:
    """Artifact of a single scheduling iteration step."""
    running_sequences: List[Sequence] = field(default_factory=list)
    preempted_sequences: List[Sequence] = field(default_factory=list)
    finished_sequences: List[Sequence] = field(default_factory=list)


class Scheduler:
    """Iteration-level continuous batcher for LLM inference."""

    def __init__(
        self,
        block_manager: BlockManager,
        max_batch_size: Optional[int] = None,
        watermark_blocks: Optional[int] = None,
    ) -> None:
        self.block_manager: BlockManager = block_manager
        self.max_batch_size: int = max_batch_size or settings.MAX_BATCH_SIZE
        self.watermark_blocks: int = watermark_blocks or settings.WATERMARK_BLOCKS

        # Sequences waiting to be admitted
        self.waiting_queue: Deque[Sequence] = collections.deque()
        # Actively executing sequences in current iteration
        self.running_sequences: List[Sequence] = []

        # Cumulative counters
        self.total_completed: int = 0
        self.total_preemptions: int = 0

    @property
    def num_waiting(self) -> int:
        """Returns the number of waiting requests."""
        return len(self.waiting_queue)

    @property
    def num_running(self) -> int:
        """Returns the number of currently running sequences."""
        return len(self.running_sequences)

    def add_sequence(self, seq: Sequence) -> None:
        """Appends a new sequence to the waiting queue."""
        if len(self.waiting_queue) >= settings.MAX_WAITING_QUEUE_SIZE:
            raise RuntimeError("Scheduler waiting queue is at maximum capacity.")

        seq.status = SequenceStatus.WAITING
        self.waiting_queue.append(seq)
        logger.debug("Sequence %s added to scheduler waiting queue. Queue depth: %d", seq.seq_id, len(self.waiting_queue))

    def has_work(self) -> bool:
        """Returns True if there are running or waiting sequences to process."""
        return bool(self.running_sequences or self.waiting_queue)

    def schedule(self) -> SchedulerOutput:
        """Executes one iteration-level scheduling step.

        1. Evicts and deallocates finished sequences.
        2. Checks memory pressure on running batch and preempts victim sequences if needed.
        3. Admits waiting sequences into the running batch up to max_batch_size and memory headroom.
        """
        output = SchedulerOutput()

        # Step 1: Evict finished sequences from previous iteration
        active_running: List[Sequence] = []
        for seq in self.running_sequences:
            if seq.is_finished:
                self.block_manager.free(seq)
                self.total_completed += 1
                output.finished_sequences.append(seq)
            else:
                active_running.append(seq)
        self.running_sequences = active_running

        # Step 2: Memory pressure evaluation and preemption
        # Estimate additional blocks required for currently running sequences in next token step
        blocks_needed_by_running = sum(
            1 for seq in self.running_sequences if self.block_manager.will_need_new_block(seq)
        )

        # While memory headroom is violated, preempt lowest-priority (most recently scheduled) sequences
        while (
            self.running_sequences
            and (self.block_manager.num_free_blocks - blocks_needed_by_running < self.watermark_blocks)
        ):
            # Select victim sequence: lowest priority (LIFO / latest arrived)
            victim = self.running_sequences.pop()
            self.block_manager.free(victim)
            victim.reset_for_preemption()

            # Re-enqueue victim at head of waiting queue (recompute strategy)
            self.waiting_queue.appendleft(victim)
            self.total_preemptions += 1
            output.preempted_sequences.append(victim)

            logger.warning(
                "Preempted sequence %s due to KV-cache memory pressure. Free blocks: %d",
                victim.seq_id,
                self.block_manager.num_free_blocks,
            )

            # Re-evaluate blocks needed
            blocks_needed_by_running = sum(
                1 for seq in self.running_sequences if self.block_manager.will_need_new_block(seq)
            )

        # Step 3: Admit waiting sequences
        preempted_ids = {seq.seq_id for seq in output.preempted_sequences}
        while self.waiting_queue and len(self.running_sequences) < self.max_batch_size:
            candidate = self.waiting_queue[0]

            # Avoid thrashing: do not immediately re-admit sequences preempted in the same iteration
            if candidate.seq_id in preempted_ids:
                break

            # Calculate blocks needed for prompt tokens
            prompt_blocks = candidate.num_blocks_needed(self.block_manager.block_size)

            # Ensure allocating candidate does not breach the watermark reserve
            if (
                self.block_manager.can_allocate(candidate)
                and (self.block_manager.num_free_blocks - prompt_blocks >= self.watermark_blocks)
            ):
                seq = self.waiting_queue.popleft()
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING

                if seq.metrics.first_scheduled_time is None:
                    seq.metrics.first_scheduled_time = time.time()

                self.running_sequences.append(seq)
                logger.debug("Admitted sequence %s to running batch. Active batch size: %d", seq.seq_id, len(self.running_sequences))
            else:
                # KV memory insufficient for next waiting candidate; stop admitting to preserve headroom
                break

        output.running_sequences = list(self.running_sequences)
        return output
