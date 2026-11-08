"""Sequence state representation and lifecycle tracking."""

import asyncio
import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SequenceStatus(str, Enum):
    """Lifecycle states of an inference sequence."""
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    SWAPPED = "SWAPPED"
    FINISHED = "FINISHED"


class SequenceMetrics(BaseModel):
    """Detailed latency and throughput metrics for an individual sequence."""
    arrival_time: float = Field(default_factory=time.time)
    first_scheduled_time: Optional[float] = None
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    ttft_ms: Optional[float] = None  # Time-to-First-Token in milliseconds
    itl_ms_list: List[float] = Field(default_factory=list)  # Inter-Token Latencies
    avg_itl_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    preemption_count: int = 0

    def record_first_token(self, timestamp: float) -> None:
        """Records the timestamp of the first generated token and calculates TTFT."""
        self.first_token_time = timestamp
        self.last_token_time = timestamp
        self.ttft_ms = (timestamp - self.arrival_time) * 1000.0

    def record_subsequent_token(self, timestamp: float) -> None:
        """Records an inter-token step and tracks ITL."""
        if self.last_token_time is not None:
            itl = (timestamp - self.last_token_time) * 1000.0
            self.itl_ms_list.append(itl)
        self.last_token_time = timestamp

    def mark_finished(self, timestamp: float, output_token_count: int) -> None:
        """Finalizes sequence metrics upon completion."""
        self.finish_time = timestamp
        self.total_latency_ms = (timestamp - self.arrival_time) * 1000.0

        if self.itl_ms_list:
            self.avg_itl_ms = sum(self.itl_ms_list) / len(self.itl_ms_list)
        elif self.ttft_ms is not None:
            self.avg_itl_ms = 0.0
        else:
            self.avg_itl_ms = 0.0

        if self.total_latency_ms > 0 and output_token_count > 0:
            self.tokens_per_second = (output_token_count / self.total_latency_ms) * 1000.0
        else:
            self.tokens_per_second = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes metrics with clean rounding."""
        return {
            "arrival_time": self.arrival_time,
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "avg_itl_ms": round(self.avg_itl_ms, 2) if self.avg_itl_ms is not None else None,
            "total_latency_ms": round(self.total_latency_ms, 2) if self.total_latency_ms is not None else None,
            "tokens_per_second": round(self.tokens_per_second, 2) if self.tokens_per_second is not None else None,
            "preemption_count": self.preemption_count,
        }


class Sequence:
    """Represents a generation request undergoing iteration-level continuous batching."""

    def __init__(
        self,
        seq_id: str,
        prompt: str,
        prompt_tokens: List[int],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
    ) -> None:
        self.seq_id: str = seq_id
        self.prompt: str = prompt
        self.prompt_tokens: List[int] = list(prompt_tokens)
        self.output_tokens: List[int] = []
        self.output_text: str = ""
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.stop_sequences: List[str] = stop_sequences or []

        self.num_prompt_tokens_processed: int = 0
        self.status: SequenceStatus = SequenceStatus.WAITING
        self.finish_reason: Optional[str] = None
        self.metrics: SequenceMetrics = SequenceMetrics()

        # Async coordination primitives
        self.token_queue: asyncio.Queue = asyncio.Queue()
        self.completion_event: asyncio.Event = asyncio.Event()

    @property
    def num_prompt_tokens(self) -> int:
        """Returns prompt token count."""
        return len(self.prompt_tokens)

    @property
    def num_output_tokens(self) -> int:
        """Returns generated output token count."""
        return len(self.output_tokens)

    @property
    def total_tokens(self) -> int:
        """Returns combined prompt and output tokens."""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        """Returns True if the sequence has completed generation."""
        return self.status == SequenceStatus.FINISHED

    @property
    def is_prefill_complete(self) -> bool:
        """Returns True if all prompt tokens have been processed through prefill."""
        return self.num_prompt_tokens_processed >= self.num_prompt_tokens

    def advance_prefill(self, count: int) -> int:
        """Advances prompt token processing in chunked prefill mode."""
        tokens_to_process = min(count, self.num_prompt_tokens - self.num_prompt_tokens_processed)
        self.num_prompt_tokens_processed += tokens_to_process
        return tokens_to_process

    def num_blocks_needed(self, block_size: int) -> int:
        """Computes physical blocks required for the current sequence length."""
        if self.total_tokens == 0:
            return 0
        return math.ceil(self.total_tokens / block_size)

    def append_token(self, token_id: int, token_text: str, timestamp: float) -> None:
        """Appends a newly generated token and updates timing metrics."""
        self.output_tokens.append(token_id)
        self.output_text += token_text

        if len(self.output_tokens) == 1:
            self.metrics.record_first_token(timestamp)
        else:
            self.metrics.record_subsequent_token(timestamp)

    def mark_finished(self, reason: str, timestamp: float) -> None:
        """Marks the sequence as finished and triggers completion event."""
        self.status = SequenceStatus.FINISHED
        self.finish_reason = reason
        self.metrics.mark_finished(timestamp, self.num_output_tokens)
        self.completion_event.set()

    def reset_for_preemption(self) -> None:
        """Resets execution state when preempted due to KV-cache memory pressure."""
        self.status = SequenceStatus.PREEMPTED
        self.metrics.preemption_count += 1

