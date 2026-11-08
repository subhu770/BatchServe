"""BatchServe Inference Engine.

Coordinates iteration-level continuous batch execution, dynamic KV block
allocation, logits computation, nucleus/temperature sampling, and token streaming.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional
import numpy as np

from app.core.config import settings
from app.core.logger import logger
from app.models.schemas import CompletionRequest, MetricsResponse, StreamChunk
from app.models.sequence import Sequence, SequenceStatus
from app.services.block_manager import BlockManager
from app.services.scheduler import Scheduler
from app.services.tokenizer import SimpleTokenizer, tokenizer as default_tokenizer


class InferenceEngine:
    """Core LLM continuous batch serving engine."""

    def __init__(
        self,
        block_manager: Optional[BlockManager] = None,
        scheduler: Optional[Scheduler] = None,
        tokenizer: Optional[SimpleTokenizer] = None,
    ) -> None:
        self.tokenizer: SimpleTokenizer = tokenizer or default_tokenizer
        self.block_manager: BlockManager = block_manager or BlockManager()
        self.scheduler: Scheduler = scheduler or Scheduler(block_manager=self.block_manager)

        # Engine execution state
        self._is_running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self._work_event: asyncio.Event = asyncio.Event()

        # Engine telemetry
        self.start_time: float = time.time()
        self.total_tokens_generated: int = 0
        self.ttft_history: List[float] = []
        self.itl_history: List[float] = []

    async def start(self) -> None:
        """Starts the background continuous batching execution loop."""
        if self._is_running:
            return
        self._is_running = True
        self._loop_task = asyncio.create_task(self._step_loop())
        logger.info(
            "BatchServe Inference Engine started. Block size: %d, Total blocks: %d, Max batch: %d",
            self.block_manager.block_size,
            self.block_manager.total_blocks,
            self.scheduler.max_batch_size,
        )

    async def stop(self) -> None:
        """Gracefully halts the background loop and clears pending tasks."""
        self._is_running = False
        self._work_event.set()
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("BatchServe Inference Engine halted.")

    def add_request(self, request: CompletionRequest) -> Sequence:
        """Enqueues an incoming client request into the continuous batch scheduler."""
        seq_id = str(uuid.uuid4())
        prompt_tokens = self.tokenizer.encode(request.prompt)

        # Safeguard: ensure prompt has at least one token
        if not prompt_tokens:
            prompt_tokens = [self.tokenizer.BOS_TOKEN_ID]

        sequence = Sequence(
            seq_id=seq_id,
            prompt=request.prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop or [],
        )

        self.scheduler.add_sequence(sequence)
        self._work_event.set()
        return sequence

    async def _step_loop(self) -> None:
        """Continuous background batching loop executing iteration steps."""
        while self._is_running:
            try:
                if self.scheduler.has_work():
                    await self.step()
                else:
                    self._work_event.clear()
                    # Await new incoming requests or timeout
                    try:
                        await asyncio.wait_for(self._work_event.wait(), timeout=0.05)
                    except asyncio.TimeoutError:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in continuous batching iteration step: %s", e, exc_info=True)
                await asyncio.sleep(0.01)

    async def step(self) -> None:
        """Executes one continuous batch iteration across all active sequences."""
        sched_output = self.scheduler.schedule()
        running_seqs = sched_output.running_sequences

        if not running_seqs:
            # Yield to event loop if no sequences can run currently
            await asyncio.sleep(0.001)
            return

        # Simulate micro-forward pass compute latency
        if settings.ENABLE_SIMULATED_LATENCY and settings.SIMULATED_STEP_LATENCY_MS > 0:
            await asyncio.sleep(settings.SIMULATED_STEP_LATENCY_MS / 1000.0)

        step_timestamp = time.time()

        for seq in running_seqs:
            # 1. Allocate KV block if crossing block boundary
            try:
                self.block_manager.append_token(seq)
            except MemoryError:
                logger.warning("Seq %s hit KV cache limit during step. Preempting.", seq.seq_id)
                self.block_manager.free(seq)
                seq.reset_for_preemption()
                self.scheduler.waiting_queue.appendleft(seq)
                continue

            # 2. Sample next token via logits distribution
            token_id = self._sample_next_token(seq)
            token_text = self.tokenizer.decode_single(token_id)

            # 3. Append token to sequence
            seq.append_token(token_id, token_text, step_timestamp)
            self.total_tokens_generated += 1

            if len(seq.output_tokens) == 1 and seq.metrics.ttft_ms is not None:
                self.ttft_history.append(seq.metrics.ttft_ms)
                if len(self.ttft_history) > 1000:
                    self.ttft_history.pop(0)

            if seq.metrics.itl_ms_list:
                self.itl_history.append(seq.metrics.itl_ms_list[-1])
                if len(self.itl_history) > 1000:
                    self.itl_history.pop(0)

            # 4. Check termination criteria
            finish_reason = self._check_finish(seq, token_id)

            if finish_reason:
                seq.mark_finished(finish_reason, step_timestamp)
                # Emit final streaming chunk with completion metrics
                chunk = StreamChunk(
                    id=seq.seq_id,
                    token=token_text,
                    token_id=token_id,
                    step=len(seq.output_tokens),
                    finish_reason=finish_reason,
                    metrics=seq.metrics.to_dict(),
                )
                seq.token_queue.put_nowait(chunk)
                # Sentinel to signal stream EOF
                seq.token_queue.put_nowait(None)
            else:
                # Emit intermediate streaming token chunk
                chunk = StreamChunk(
                    id=seq.seq_id,
                    token=token_text,
                    token_id=token_id,
                    step=len(seq.output_tokens),
                    finish_reason=None,
                )
                seq.token_queue.put_nowait(chunk)

    def _sample_next_token(self, seq: Sequence) -> int:
        """Samples next token using temperature scaling and nucleus (top-p) sampling."""
        vocab_size = self.tokenizer.vocab_size

        # Construct realistic context-aware pseudo-logits
        last_tok = seq.output_tokens[-1] if seq.output_tokens else seq.prompt_tokens[-1]
        rng_seed = (hash(seq.seq_id) + len(seq.output_tokens) * 31 + last_tok * 17) % (2**31 - 1)
        step_rng = np.random.RandomState(rng_seed)

        # Base distribution over vocabulary
        raw_logits = step_rng.normal(loc=0.0, scale=1.0, size=vocab_size)

        # Favor word tokens over punctuation/special tokens
        raw_logits[4:150] += 1.5

        # Small probability for EOS token if sequence has produced tokens
        if len(seq.output_tokens) >= 8:
            raw_logits[SimpleTokenizer.EOS_TOKEN_ID] += 0.8
        else:
            raw_logits[SimpleTokenizer.EOS_TOKEN_ID] = -1e9  # Suppress early EOS

        # Temperature scaling
        temperature = max(seq.temperature, 1e-4)
        scaled_logits = raw_logits / temperature

        # Numerical stability shift
        scaled_logits -= np.max(scaled_logits)
        exp_logits = np.exp(scaled_logits)
        probs = exp_logits / np.sum(exp_logits)

        # Top-p (Nucleus) filtering
        top_p = min(max(seq.top_p, 0.01), 1.0)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Find cutoff
        cutoff_mask = cumulative_probs > top_p
        # Shift mask right so at least the first token above top_p is included
        cutoff_mask[1:] = cutoff_mask[:-1]
        cutoff_mask[0] = False

        filtered_indices = sorted_indices[~cutoff_mask]
        filtered_probs = sorted_probs[~cutoff_mask]

        if len(filtered_indices) == 0:
            return int(sorted_indices[0])

        filtered_probs /= np.sum(filtered_probs)
        sampled_token_id = int(np.random.choice(filtered_indices, p=filtered_probs))
        return sampled_token_id

    def _check_finish(self, seq: Sequence, last_token_id: int) -> Optional[str]:
        """Evaluates whether the sequence should terminate."""
        if last_token_id == SimpleTokenizer.EOS_TOKEN_ID:
            return "stop"

        if len(seq.output_tokens) >= seq.max_new_tokens:
            return "length"

        for stop_seq in seq.stop_sequences:
            if stop_seq and stop_seq in seq.output_text:
                return "stop"

        return None

    def get_metrics(self) -> MetricsResponse:
        """Gathers aggregate server and KV-cache telemetry."""
        now = time.time()
        elapsed = max(now - self.start_time, 0.001)
        tokens_per_sec = self.total_tokens_generated / elapsed

        avg_ttft = float(np.mean(self.ttft_history)) if self.ttft_history else 0.0
        avg_itl = float(np.mean(self.itl_history)) if self.itl_history else 0.0

        return MetricsResponse(
            timestamp=now,
            cache_hit_rate=round(self.block_manager.cache_hit_rate, 4),
            total_blocks=self.block_manager.total_blocks,
            free_blocks=self.block_manager.num_free_blocks,
            used_blocks=self.block_manager.num_used_blocks,
            kv_cache_utilization_pct=round(self.block_manager.kv_cache_utilization * 100.0, 2),
            active_sequences=self.scheduler.num_running,
            waiting_sequences=self.scheduler.num_waiting,
            preempted_sequences=self.scheduler.total_preemptions,
            total_completed_sequences=self.scheduler.total_completed,
            total_tokens_generated=self.total_tokens_generated,
            avg_ttft_ms=round(avg_ttft, 2),
            avg_itl_ms=round(avg_itl, 2),
            tokens_per_second=round(tokens_per_sec, 2),
        )


# Global engine singleton
engine = InferenceEngine()

