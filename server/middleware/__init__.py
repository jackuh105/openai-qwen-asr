import asyncio
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from server.config import config


class ConcurrencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in (
            "/health",
            "/v1/models",
            "/metrics",
            "/docs",
            "/openapi.json",
        ):
            return await call_next(request)

        semaphore: asyncio.Semaphore = request.app.state.semaphore

        if semaphore.locked() and semaphore._value == 0:
            from server.errors import server_busy_error
            from fastapi.responses import JSONResponse

            error = server_busy_error(config.max_concurrent_requests)
            return JSONResponse(
                status_code=503,
                content=error.to_response(),
            )

        async with semaphore:
            return await call_next(request)
