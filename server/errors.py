from typing import Optional
from pydantic import BaseModel
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class OpenAIError(Exception):
    def __init__(
        self,
        message: str,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
        status_code: int = 400,
    ):
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code
        self.status_code = status_code
        super().__init__(message)

    def to_response(self) -> dict:
        return ErrorResponse(
            error=ErrorDetail(
                message=self.message,
                type=self.error_type,
                param=self.param,
                code=self.code,
            )
        ).model_dump()


def invalid_file_error(message: str = "Unsupported audio file format") -> OpenAIError:
    return OpenAIError(
        message=message,
        error_type="invalid_request_error",
        param="file",
        code="invalid_file",
        status_code=400,
    )


def file_too_large_error(max_size_mb: int) -> OpenAIError:
    return OpenAIError(
        message=f"File size exceeds maximum allowed size of {max_size_mb}MB",
        error_type="invalid_request_error",
        param="file",
        code="file_too_large",
        status_code=400,
    )


def invalid_model_error(model: str) -> OpenAIError:
    return OpenAIError(
        message=f"The model '{model}' does not exist",
        error_type="invalid_request_error",
        param="model",
        code="invalid_model",
        status_code=400,
    )


def invalid_response_format_error(format_name: str) -> OpenAIError:
    return OpenAIError(
        message=f"Invalid response format: '{format_name}'. Supported formats: json, text, srt, vtt, verbose_json",
        error_type="invalid_request_error",
        param="response_format",
        code="invalid_response_format",
        status_code=400,
    )


def transcription_failed_error(message: str = "Transcription failed") -> OpenAIError:
    return OpenAIError(
        message=message,
        error_type="server_error",
        code="transcription_failed",
        status_code=500,
    )


def server_busy_error() -> OpenAIError:
    return OpenAIError(
        message="Server is busy. Maximum concurrent requests reached.",
        error_type="server_error",
        code="server_busy",
        status_code=503,
    )


async def openai_error_handler(request: Request, exc: OpenAIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_response())


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(message=str(exc.detail), type="invalid_request_error")
        ).model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                message="Internal server error",
                type="server_error",
                code="internal_error",
            )
        ).model_dump(),
    )
