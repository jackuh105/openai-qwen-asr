from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from server.config import config
from server.models import TranscriptionResponse, VerboseJsonResponse, ResponseFormat
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

        result = ASREngine.transcribe(
            audio, language=language, return_timestamps=return_timestamps
        )

        text = result.get("text", "")

        if response_format == "json":
            return TranscriptionResponse(text=text).model_dump()

        elif response_format == "text":
            return PlainTextResponse(content=text)

        elif response_format == "srt":
            segments = result.get("segments", [])
            if not segments and text:
                segments = [{"start": 0.0, "end": duration, "text": text}]
            srt_content = segments_to_srt(segments)
            return PlainTextResponse(content=srt_content, media_type="text/plain")

        elif response_format == "vtt":
            segments = result.get("segments", [])
            if not segments and text:
                segments = [{"start": 0.0, "end": duration, "text": text}]
            vtt_content = segments_to_vtt(segments)
            return PlainTextResponse(content=vtt_content, media_type="text/vtt")

        elif response_format == "verbose_json":
            segments = result.get("segments", [])
            words = []
            for seg in segments:
                if "words" in seg:
                    words.extend(seg["words"])

            return VerboseJsonResponse(
                task="transcribe",
                language=result.get("language", language),
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
