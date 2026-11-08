"""Pydantic v2 schemas for BatchServe API request and response contracts."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """Client request schema for text generation."""
    prompt: str = Field(
        ...,
        min_length=1,
        description="The prompt string to generate completions for.",
        examples=["Explain how paged attention eliminates external memory fragmentation."],
    )
    max_tokens: int = Field(
        default=32,
        ge=1,
        le=2048,
        description="The maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values yield more diverse outputs.",
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold.",
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Optional list of stop sequences that halt generation.",
    )


class CompletionResponse(BaseModel):
    """Response schema for non-streaming batch generation."""
    id: str = Field(..., description="Unique request identifier")
    prompt: str = Field(..., description="Input prompt text")
    text: str = Field(..., description="Complete generated text")
    output_tokens: List[int] = Field(..., description="List of generated token IDs")
    finish_reason: str = Field(..., description="Reason for termination: 'length', 'stop', etc.")
    metrics: Dict[str, Any] = Field(..., description="Latency and performance telemetry")


class StreamChunk(BaseModel):
    """Individual Server-Sent Event (SSE) chunk emitted per iteration step."""
    id: str = Field(..., description="Request identifier")
    token: str = Field(..., description="Newly generated token string")
    token_id: Optional[int] = Field(default=None, description="Generated token ID")
    step: int = Field(..., description="Generation iteration step number")
    finish_reason: Optional[str] = Field(default=None, description="Reason if sequence finished")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Final metrics on last chunk")


class MetricsResponse(BaseModel):
    """Engine telemetry and performance metrics."""
    timestamp: float = Field(..., description="Current server epoch timestamp")
    cache_hit_rate: float = Field(..., description="Prefix KV-cache hit rate [0.0 - 1.0]")
    total_blocks: int = Field(..., description="Total physical memory blocks in pool")
    free_blocks: int = Field(..., description="Available physical memory blocks")
    used_blocks: int = Field(..., description="Currently allocated physical blocks")
    kv_cache_utilization_pct: float = Field(..., description="KV-cache memory utilization percentage")
    active_sequences: int = Field(..., description="Sequences currently executing in iteration batch")
    waiting_sequences: int = Field(..., description="Sequences queued in scheduler")
    preempted_sequences: int = Field(..., description="Total preempted sequence occurrences")
    total_completed_sequences: int = Field(..., description="Total successfully served sequences")
    total_tokens_generated: int = Field(..., description="Cumulative tokens produced across all sequences")
    avg_ttft_ms: float = Field(..., description="Rolling average Time-To-First-Token in ms")
    avg_itl_ms: float = Field(..., description="Rolling average Inter-Token Latency in ms")
    tokens_per_second: float = Field(..., description="Engine output throughput in tokens/sec")


class HealthResponse(BaseModel):
    """Server health status."""
    status: str = Field(default="healthy")
    engine: str = Field(default="BatchServe")
    version: str = Field(...)
    total_blocks: int = Field(...)
    free_blocks: int = Field(...)
    active_sequences: int = Field(...)

