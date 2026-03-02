from typing import Optional, TYPE_CHECKING, AsyncIterator
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse

from server.config import config
from server.models import TranscriptionResponse, VerboseJsonResponse, ResponseFormat
from server.models import TranscriptPartialEvent, TranscriptFinalEvent
from server.errors import (
    OpenAIError,
    invalid_file_error,
    file_too_large_error,
    transcription_failed_error,
)
from server.utils.model_mapping import resolve_model
from server.utils.audio import (
    load_audio_from_bytes,
    get_audio_duration,
    segments_to_srt,
    segments_to_vtt,
)
from server.asr.engine import ASREngine
from server.asr.streaming import StreamingTranscriber

if TYPE_CHECKING:
    from mlx_qwen3_asr import TranscriptionResult


router = APIRouter(tags=["Audio"])


SUPPORTED_AUDIO_TYPES = {
    "audio/wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/m4a",
    "audio/x-m4a",
    "audio/flac",
    "audio/ogg",
    "audio/webm",
    "video/mp4",
    "video/webm",
}


def validate_file(file: UploadFile) -> None:
    content_type = file.content_type or ""

    if content_type not in SUPPORTED_AUDIO_TYPES:
        filename = file.filename or "unknown"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in {"wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4"}:
            raise invalid_file_error(
                f"Unsupported file type: {content_type or filename}"
            )


@router.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    response_format: ResponseFormat = Form(default="json"),
    temperature: Optional[float] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    stream: bool = Form(default=False),
):
    validate_file(file)

    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > config.max_file_size_mb:
        raise file_too_large_error(config.max_file_size_mb)

    resolved_model = resolve_model(model)

    return_timestamps = response_format in ("srt", "vtt", "verbose_json")

    try:
        audio, sample_rate = load_audio_from_bytes(file_bytes)
        duration = get_audio_duration(audio, sample_rate)

        if stream:
            return StreamingResponse(
                _generate_sse_stream(audio, language, response_format),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        result = ASREngine.transcribe(
            audio, language=language, return_timestamps=return_timestamps
        )

        text = result.text

        if response_format == "json":
            return TranscriptionResponse(text=text).model_dump()

        elif response_format == "text":
            return PlainTextResponse(content=text)

        elif response_format == "srt":
            segments = result.segments or []
            if not segments and text:
                segments = [{"start": 0.0, "end": duration, "text": text}]
            srt_content = segments_to_srt(segments)
            return PlainTextResponse(content=srt_content, media_type="text/plain")

        elif response_format == "vtt":
            segments = result.segments or []
            if not segments and text:
                segments = [{"start": 0.0, "end": duration, "text": text}]
            vtt_content = segments_to_vtt(segments)
            return PlainTextResponse(content=vtt_content, media_type="text/vtt")

        elif response_format == "verbose_json":
            segments = result.segments or []
            words = []
            for seg in segments:
                if "words" in seg:
                    words.extend(seg["words"])

            return VerboseJsonResponse(
                task="transcribe",
                language=result.language or language,
                duration=duration,
                text=text,
                words=words if words else None,
            ).model_dump()

        else:
            return TranscriptionResponse(text=text).model_dump()

    except OpenAIError:
        raise
    except Exception as e:
        raise transcription_failed_error(str(e))


async def _generate_sse_stream(
    audio, language: Optional[str], response_format: ResponseFormat
) -> AsyncIterator[str]:
    transcriber = StreamingTranscriber(config)

    async for event_type, text in transcriber.transcribe_stream_with_deltas(
        audio, language=language
    ):
        if event_type == "partial":
            event = TranscriptPartialEvent(text=text)
        else:
            event = TranscriptFinalEvent(text=text)

        yield f"event: {event.type}\n"
        yield f"data: {event.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"
