import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from server.config import config
from server.errors import (
    OpenAIError,
    openai_error_handler,
    http_exception_handler,
    generic_exception_handler,
)
from server.asr.engine import ASREngine
from server.routes.transcriptions import router as transcriptions_router


request_semaphore: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global request_semaphore
    request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    print(f"Loading model: {config.model_id}")
    ASREngine.get_instance(config)
    ASREngine.load_model()
    print("Model loaded successfully")

    yield

    print("Shutting down...")


app = FastAPI(
    title="OpenAI-compatible ASR Server",
    description="OpenAI API compatible ASR server using mlx-qwen3-asr",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_exception_handler(OpenAIError, openai_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)


app.include_router(transcriptions_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": ASREngine.is_loaded()}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "whisper-1", "object": "model", "owned_by": "openai"},
            {"id": "Qwen/Qwen3-ASR-0.6B", "object": "model", "owned_by": "qwen"},
            {"id": "Qwen/Qwen3-ASR-1.7B", "object": "model", "owned_by": "qwen"},
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host=config.host, port=config.port, reload=False)
