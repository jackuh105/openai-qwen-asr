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
