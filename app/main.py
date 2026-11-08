"""BatchServe Application Entrypoint.

Configures FastAPI, registers API routes, manages application lifespan
(starting/stopping the continuous batching inference engine), and handles exceptions.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.core.logger import logger
from app.services.engine import engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler managing inference engine background tasks."""
    logger.info("Initializing %s v%s...", settings.PROJECT_NAME, settings.VERSION)
    await engine.start()
    yield
    logger.info("Shutting down %s...", settings.PROJECT_NAME)
    await engine.stop()


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI instance."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Production-grade lightweight LLM serving engine featuring "
            "iteration-level continuous batching and Paged KV-Cache memory allocation."
        ),
        lifespan=lifespan,
    )

    # CORS configuration for client dashboards and streaming
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    application.include_router(router)

    @application.get("/", tags=["General"])
    async def root_info() -> JSONResponse:
        """Root endpoint describing BatchServe capabilities."""
        return JSONResponse(
            content={
                "name": settings.PROJECT_NAME,
                "version": settings.VERSION,
                "status": "online",
                "docs": "/docs",
                "endpoints": {
                    "generate": f"{settings.API_V1_STR}/generate",
                    "stream": f"{settings.API_V1_STR}/stream",
                    "metrics": "/metrics",
                    "health": "/health",
                },
                "kv_cache_config": {
                    "block_size": settings.BLOCK_SIZE,
                    "total_blocks": settings.TOTAL_BLOCKS,
                    "prefix_caching": settings.ENABLE_PREFIX_CACHING,
                },
            }
        )

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

