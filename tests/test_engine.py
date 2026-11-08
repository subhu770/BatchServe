"""Tests for the Inference Engine and generation step loop."""

import asyncio
import pytest
from app.models.schemas import CompletionRequest
from app.services.engine import InferenceEngine


@pytest.mark.asyncio
async def test_engine_single_step(engine: InferenceEngine) -> None:
    """Verifies that executing a single engine step generates a token and emits chunk."""
    req = CompletionRequest(prompt="Hello continuous batching", max_tokens=4, temperature=0.7)
    seq = engine.add_request(req)

    assert seq.seq_id in engine.scheduler.waiting_queue[0].seq_id
    assert len(seq.output_tokens) == 0

    await engine.step()

    assert len(seq.output_tokens) == 1
    assert seq.metrics.first_token_time is not None
    assert seq.metrics.ttft_ms is not None

    chunk = seq.token_queue.get_nowait()
    assert chunk is not None
    assert chunk.step == 1
    assert chunk.token_id == seq.output_tokens[0]


@pytest.mark.asyncio
async def test_engine_completion_lifecycle(engine: InferenceEngine) -> None:
    """Verifies that engine steps run until max_tokens and marks sequence finished."""
    req = CompletionRequest(prompt="Explain memory blocks", max_tokens=5, temperature=0.7)
    seq = engine.add_request(req)

    for _ in range(10):
        if seq.is_finished:
            break
        await engine.step()

    assert seq.is_finished is True
    assert seq.finish_reason == "length"
    assert len(seq.output_tokens) == 5
    assert seq.completion_event.is_set()
    assert seq.metrics.total_latency_ms is not None
    assert seq.metrics.avg_itl_ms is not None
    assert seq.metrics.tokens_per_second is not None


@pytest.mark.asyncio
async def test_engine_stop_sequence(engine: InferenceEngine) -> None:
    """Verifies termination when a stop sequence is encountered."""
    # Force stop sequence on space token or letter
    req = CompletionRequest(
        prompt="Test stop word",
        max_tokens=20,
        stop=[" "],
    )
    seq = engine.add_request(req)

    for _ in range(25):
        if seq.is_finished:
            break
        await engine.step()

    assert seq.is_finished is True
    assert seq.finish_reason in ("stop", "length")


@pytest.mark.asyncio
async def test_engine_metrics_endpoint_data(engine: InferenceEngine) -> None:
    """Verifies engine aggregate telemetry calculation."""
    req = CompletionRequest(prompt="Metrics test", max_tokens=3)
    seq = engine.add_request(req)

    while not seq.is_finished:
        await engine.step()

    metrics = engine.get_metrics()
    assert metrics.total_tokens_generated >= 3
    assert metrics.total_completed_sequences >= 0
    assert metrics.free_blocks <= metrics.total_blocks

