"""BatchServe REST and Server-Sent Events (SSE) API routes."""

import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logger import logger
from app.models.schemas import (
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    MetricsResponse,
)
from app.services.engine import engine

router = APIRouter()


@router.post(
    "/api/v1/generate",
    response_model=CompletionResponse,
    status_code=status.HTTP_200_OK,
    summary="Non-streaming continuous batch generation",
    description="Submits a completion request to the continuous batch engine and returns complete generated output with full latency breakdown (TTFT, ITL).",
)
async def generate_completion(request: CompletionRequest) -> CompletionResponse:
    """Synchronous-wait completion endpoint returning full text and latency breakdown."""
    try:
        seq = engine.add_request(request)
        # Await completion event with request timeout safety
        await asyncio.wait_for(seq.completion_event.wait(), timeout=60.0)

        return CompletionResponse(
            id=seq.seq_id,
            prompt=seq.prompt,
            text=seq.output_text,
            output_tokens=seq.output_tokens,
            finish_reason=seq.finish_reason or "stop",
            metrics=seq.metrics.to_dict(),
        )
    except asyncio.TimeoutError:
        logger.error("Generation request timed out for prompt: %s", request.prompt[:50])
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request processing timed out in continuous batch scheduler.",
        )
    except Exception as e:
        logger.error("Error generating completion: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/api/v1/stream",
    status_code=status.HTTP_200_OK,
    summary="Server-Sent Events (SSE) token stream",
    description="Streams generated tokens incrementally as they are produced in each continuous batching iteration.",
)
async def stream_completion(request: CompletionRequest) -> StreamingResponse:
    """Streams generated tokens per iteration step using Server-Sent Events."""
    seq = engine.add_request(request)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                chunk = await seq.token_queue.get()
                if chunk is None:
                    # EOF marker
                    break
                yield f"data: {chunk.model_dump_json()}\n\n"
        except asyncio.CancelledError:
            logger.info("Client disconnected from SSE stream for seq %s", seq.seq_id)
        except Exception as e:
            logger.error("Error during streaming generation: %s", e)
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Engine telemetry and KV-cache utilization",
    description="Exposes real-time cache hit rate, allocated blocks, active/waiting sequences, and tokens/sec.",
)
async def get_metrics() -> MetricsResponse:
    """Returns real-time engine telemetry."""
    return engine.get_metrics()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check and block memory status",
    description="Returns server status and KV-cache capacity.",
)
async def health_check() -> HealthResponse:
    """Returns system health and block availability."""
    return HealthResponse(
        status="healthy",
        engine=settings.PROJECT_NAME,
        version=settings.VERSION,
        total_blocks=engine.block_manager.total_blocks,
        free_blocks=engine.block_manager.num_free_blocks,
        active_sequences=engine.scheduler.num_running,
    )

