from typing import Optional, List, Literal
from pydantic import BaseModel, Field, ConfigDict


ResponseFormat = Literal["json", "text", "srt", "vtt", "verbose_json"]


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionResponse(BaseModel):
    text: str


class VerboseJsonResponse(BaseModel):
    task: str = "transcribe"
    language: Optional[str] = None
    duration: Optional[float] = None
    text: str
    words: Optional[List[WordTimestamp]] = None


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="whisper-1")
    language: Optional[str] = None
    response_format: ResponseFormat = Field(default="json")
    stream: bool = Field(default=False)


class TranscriptPartialEvent(BaseModel):
    type: str = "transcript.partial"
    text: str


class TranscriptFinalEvent(BaseModel):
    type: str = "transcript.final"
    text: str


class RealtimeSessionConfig(BaseModel):
    model: str = Field(default="whisper-1")
    input_audio_format: str = Field(default="pcm16")


class RealtimeSession(BaseModel):
    id: str
    model: str


class SessionUpdateEvent(BaseModel):
    type: str = "session.update"
    session: RealtimeSessionConfig


class SessionCreatedEvent(BaseModel):
    type: str = "session.created"
    session: RealtimeSession


class SessionUpdatedEvent(BaseModel):
    type: str = "session.updated"
    session: RealtimeSession


class InputAudioBufferAppendEvent(BaseModel):
    type: str = "input_audio_buffer.append"
    audio: str


class InputAudioBufferCommitEvent(BaseModel):
    type: str = "input_audio_buffer.commit"


class InputAudioBufferClearEvent(BaseModel):
    type: str = "input_audio_buffer.clear"


class InputAudioBufferCommittedEvent(BaseModel):
    type: str = "input_audio_buffer.committed"


class InputAudioBufferSpeechStartedEvent(BaseModel):
    type: str = "input_audio_buffer.speech_started"


class InputAudioBufferSpeechStoppedEvent(BaseModel):
    type: str = "input_audio_buffer.speech_stopped"


class ResponseCreatedEvent(BaseModel):
    type: str = "response.created"
    response_id: str


class ResponseAudioTranscriptDeltaEvent(BaseModel):
    type: str = "response.audio_transcript.delta"
    response_id: str
    delta: str


class ResponseAudioTranscriptDoneEvent(BaseModel):
    type: str = "response.audio_transcript.done"
    response_id: str
    transcript: str


class ResponseDoneEvent(BaseModel):
    type: str = "response.done"
    response_id: str


class ErrorDetail(BaseModel):
    type: str
    message: str
    code: Optional[str] = None


class ErrorEvent(BaseModel):
    type: str = "error"
    error: ErrorDetail
