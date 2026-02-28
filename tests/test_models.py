import pytest
from server.models import (
    TranscriptionRequest,
    TranscriptionResponse,
    VerboseJsonResponse,
    WordTimestamp,
    ResponseFormat,
)


class TestTranscriptionRequest:
    def test_default_values(self):
        request = TranscriptionRequest()
        assert request.model == "whisper-1"
        assert request.language is None
        assert request.response_format == "json"
        assert request.stream is False

    def test_custom_values(self):
        request = TranscriptionRequest(
            model="qwen-asr-0.6b",
            language="en",
            response_format="verbose_json",
            stream=True,
        )
        assert request.model == "qwen-asr-0.6b"
        assert request.language == "en"
        assert request.response_format == "verbose_json"
        assert request.stream is True

    def test_extra_fields_ignored(self):
        request = TranscriptionRequest(
            model="whisper-1", temperature=0.5, prompt="Custom prompt"
        )
        assert request.model == "whisper-1"
        assert not hasattr(request, "temperature")
        assert not hasattr(request, "prompt")


class TestTranscriptionResponse:
    def test_basic_response(self):
        response = TranscriptionResponse(text="Hello world")
        assert response.text == "Hello world"

    def test_model_dump(self):
        response = TranscriptionResponse(text="Test transcription")
        data = response.model_dump()
        assert data == {"text": "Test transcription"}


class TestVerboseJsonResponse:
    def test_minimal_response(self):
        response = VerboseJsonResponse(text="Hello")
        assert response.task == "transcribe"
        assert response.text == "Hello"
        assert response.language is None
        assert response.duration is None
        assert response.words is None

    def test_full_response(self):
        words = [
            WordTimestamp(word="Hello", start=0.0, end=0.5),
            WordTimestamp(word="world", start=0.6, end=1.0),
        ]
        response = VerboseJsonResponse(
            task="transcribe",
            language="english",
            duration=5.2,
            text="Hello world",
            words=words,
        )
        assert response.language == "english"
        assert response.duration == 5.2
        assert len(response.words) == 2
        assert response.words[0].word == "Hello"


class TestWordTimestamp:
    def test_word_timestamp(self):
        ts = WordTimestamp(word="test", start=1.0, end=1.5)
        assert ts.word == "test"
        assert ts.start == 1.0
        assert ts.end == 1.5


class TestResponseFormat:
    def test_valid_formats(self):
        valid_formats = ["json", "text", "srt", "vtt", "verbose_json"]
        for fmt in valid_formats:
            request = TranscriptionRequest(response_format=fmt)
            assert request.response_format == fmt
