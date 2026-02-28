import pytest
from server.utils.audio import (
    format_timestamp,
    format_timestamp_vtt,
    segments_to_srt,
    segments_to_vtt,
    get_audio_duration,
)


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_seconds_only(self):
        assert format_timestamp(5.5) == "00:00:05,500"

    def test_minutes_and_seconds(self):
        assert format_timestamp(65.123) == "00:01:05,123"

    def test_hours(self):
        assert format_timestamp(3661.5) == "01:01:01,500"

    def test_milliseconds(self):
        assert format_timestamp(1.999) == "00:00:01,999"


class TestFormatTimestampVtt:
    def test_zero(self):
        assert format_timestamp_vtt(0.0) == "00:00:00.000"

    def test_seconds_only(self):
        assert format_timestamp_vtt(5.5) == "00:00:05.500"

    def test_uses_dot_not_comma(self):
        result = format_timestamp_vtt(65.123)
        assert "." in result
        assert "," not in result


class TestSegmentsToSrt:
    def test_single_segment(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
        result = segments_to_srt(segments)
        assert "1" in result
        assert "00:00:00,000 --> 00:00:01,000" in result
        assert "Hello world" in result

    def test_multiple_segments(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.5, "end": 2.5, "text": "World"},
        ]
        result = segments_to_srt(segments)
        lines = result.strip().split("\n")
        assert "1" in lines[0]
        assert "2" in lines[4]

    def test_empty_segments(self):
        result = segments_to_srt([])
        assert result == ""


class TestSegmentsToVtt:
    def test_header(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Test"}]
        result = segments_to_vtt(segments)
        assert result.startswith("WEBVTT")

    def test_single_segment(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        result = segments_to_vtt(segments)
        assert "00:00:00.000 --> 00:00:01.000" in result
        assert "Hello" in result

    def test_empty_segments(self):
        result = segments_to_vtt([])
        assert result == "WEBVTT\n"


class TestGetAudioDuration:
    def test_duration_calculation(self):
        import numpy as np

        audio = np.zeros(16000)
        duration = get_audio_duration(audio, 16000)
        assert duration == 1.0

    def test_half_second(self):
        import numpy as np

        audio = np.zeros(8000)
        duration = get_audio_duration(audio, 16000)
        assert duration == 0.5

    def test_custom_sample_rate(self):
        import numpy as np

        audio = np.zeros(44100)
        duration = get_audio_duration(audio, 44100)
        assert duration == 1.0
