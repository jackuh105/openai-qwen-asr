import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from server.asr.streaming import StreamingTranscriber
from server.config import ServerConfig
from server.models import TranscriptPartialEvent, TranscriptFinalEvent


class TestStreamingTranscriber:
    def test_init(self):
        config = ServerConfig(chunk_size_sec=2.0, max_context_sec=30.0)
        transcriber = StreamingTranscriber(config)
        assert transcriber.config == config

    @patch("mlx_qwen3_asr.streaming.init_streaming")
    @patch("mlx_qwen3_asr.streaming._ModelHolder")
    def test_init_state(self, mock_holder, mock_init):
        config = ServerConfig(model_id="test-model")
        transcriber = StreamingTranscriber(config)

        mock_model = Mock()
        mock_holder.get.return_value = (mock_model, None)
        mock_state = Mock()
        mock_init.return_value = mock_state

        state = transcriber.init_state(language="en")

        mock_holder.get.assert_called_once()
        mock_init.assert_called_once()
        assert state == mock_state
        assert transcriber._model_obj == mock_model

    @patch("mlx_qwen3_asr.streaming.feed_audio")
    @patch("mlx_qwen3_asr.streaming.init_streaming")
    @patch("mlx_qwen3_asr.streaming._ModelHolder")
    def test_feed_audio(self, mock_holder, mock_init, mock_feed):
        config = ServerConfig()
        transcriber = StreamingTranscriber(config)

        mock_model = Mock()
        mock_holder.get.return_value = (mock_model, None)
        mock_init.return_value = Mock()

        transcriber.init_state()

        audio = np.zeros(16000, dtype=np.float32)
        state = Mock()
        new_state = Mock()
        mock_feed.return_value = new_state

        result = transcriber.feed_audio(audio, state)

        mock_feed.assert_called_once_with(audio, state, model=mock_model)
        assert result == new_state

    @patch("mlx_qwen3_asr.streaming.finish_streaming")
    @patch("mlx_qwen3_asr.streaming.init_streaming")
    @patch("mlx_qwen3_asr.streaming._ModelHolder")
    def test_finish(self, mock_holder, mock_init, mock_finish):
        config = ServerConfig()
        transcriber = StreamingTranscriber(config)

        mock_model = Mock()
        mock_holder.get.return_value = (mock_model, None)
        mock_init.return_value = Mock()

        transcriber.init_state()

        state = Mock()
        final_state = Mock()
        mock_finish.return_value = final_state

        result = transcriber.finish(state)

        mock_finish.assert_called_once_with(state, model=mock_model)
        assert result == final_state


class TestTranscriptEvents:
    def test_partial_event(self):
        event = TranscriptPartialEvent(text="Hello")
        assert event.type == "transcript.partial"
        assert event.text == "Hello"
        data = event.model_dump()
        assert data["type"] == "transcript.partial"
        assert data["text"] == "Hello"

    def test_final_event(self):
        event = TranscriptFinalEvent(text="Hello world.")
        assert event.type == "transcript.final"
        assert event.text == "Hello world."

    def test_event_json_serialization(self):
        event = TranscriptPartialEvent(text="Test")
        json_str = event.model_dump_json()
        assert '"type":"transcript.partial"' in json_str
        assert '"text":"Test"' in json_str


class TestStreamingIntegration:
    @pytest.mark.asyncio
    async def test_transcribe_stream_yields_text(self):
        config = ServerConfig(chunk_size_sec=0.5, max_context_sec=10.0)

        with (
            patch("mlx_qwen3_asr.streaming._ModelHolder") as mock_holder,
            patch("mlx_qwen3_asr.streaming.init_streaming") as mock_init,
            patch("mlx_qwen3_asr.streaming.feed_audio") as mock_feed,
            patch("mlx_qwen3_asr.streaming.finish_streaming") as mock_finish,
        ):
            mock_model = Mock()
            mock_holder.get.return_value = (mock_model, None)

            state1 = Mock(text="")
            state2 = Mock(text="Hello")
            state3 = Mock(text="Hello world")
            final_state = Mock(text="Hello world.")

            mock_init.return_value = state1
            mock_feed.side_effect = [state2, state3]
            mock_finish.return_value = final_state

            transcriber = StreamingTranscriber(config)
            audio = np.zeros(16000, dtype=np.float32)

            results = []
            async for text in transcriber.transcribe_stream(audio, language="en"):
                results.append(text)

            assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_transcribe_stream_with_deltas(self):
        config = ServerConfig(chunk_size_sec=0.5, max_context_sec=10.0)

        with (
            patch("mlx_qwen3_asr.streaming._ModelHolder") as mock_holder,
            patch("mlx_qwen3_asr.streaming.init_streaming") as mock_init,
            patch("mlx_qwen3_asr.streaming.feed_audio") as mock_feed,
            patch("mlx_qwen3_asr.streaming.finish_streaming") as mock_finish,
        ):
            mock_model = Mock()
            mock_holder.get.return_value = (mock_model, None)

            state1 = Mock(text="")
            state2 = Mock(text="Hello")
            final_state = Mock(text="Hello world.")

            mock_init.return_value = state1
            mock_feed.return_value = state2
            mock_finish.return_value = final_state

            transcriber = StreamingTranscriber(config)
            audio = np.zeros(16000, dtype=np.float32)

            results = []
            async for event_type, text in transcriber.transcribe_stream_with_deltas(
                audio, language="en"
            ):
                results.append((event_type, text))

            assert len(results) >= 1
            assert results[-1][0] == "final"
