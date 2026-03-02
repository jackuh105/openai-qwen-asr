import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import base64
import json
import numpy as np

from server.asr.realtime import RealtimeSessionState, RealtimeTranscriber
from server.config import ServerConfig
from server.models import (
    SessionUpdateEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    ResponseCreatedEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseDoneEvent,
    ErrorEvent,
)


class TestRealtimeSessionState:
    def test_init(self):
        config = ServerConfig()
        state = RealtimeSessionState(config)
        assert state.config == config
        assert state.session_id.startswith("sess_")
        assert state.current_text == ""
        assert state.streaming_state is None

    def test_new_response_id(self):
        config = ServerConfig()
        state = RealtimeSessionState(config)
        response_id = state.new_response_id()
        assert response_id.startswith("resp_")
        assert state.response_id == response_id

    @patch("mlx_qwen3_asr.streaming.init_streaming")
    @patch("mlx_qwen3_asr.streaming._ModelHolder")
    def test_start(self, mock_holder, mock_init):
        config = ServerConfig(model_id="test-model")
        state = RealtimeSessionState(config)

        mock_model = Mock()
        mock_holder.get.return_value = (mock_model, None)
        mock_streaming_state = Mock()
        mock_init.return_value = mock_streaming_state

        state.start()

        mock_holder.get.assert_called_once()
        mock_init.assert_called_once()
        assert state.streaming_state == mock_streaming_state
        assert state._transcriber is not None

    def test_append_audio_without_start_raises(self):
        config = ServerConfig()
        state = RealtimeSessionState(config)

        with pytest.raises(RuntimeError, match="Session not started"):
            state.append_audio(base64.b64encode(b"test").decode())

    def test_commit_without_start_raises(self):
        config = ServerConfig()
        state = RealtimeSessionState(config)

        with pytest.raises(RuntimeError, match="Session not started"):
            state.commit()

    @patch("mlx_qwen3_asr.streaming.init_streaming")
    @patch("mlx_qwen3_asr.streaming._ModelHolder")
    @patch("mlx_qwen3_asr.streaming.feed_audio")
    def test_append_audio_returns_delta(self, mock_feed, mock_holder, mock_init):
        config = ServerConfig()
        state = RealtimeSessionState(config)

        mock_model = Mock()
        mock_holder.get.return_value = (mock_model, None)
        mock_streaming_state = Mock(text="Hello world")
        mock_init.return_value = Mock(text="")
        mock_feed.return_value = mock_streaming_state

        state.start()

        audio_data = np.zeros(1600, dtype=np.float32)
        audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()
        base64_audio = base64.b64encode(audio_bytes).decode()

        delta = state.append_audio(base64_audio)

        assert delta == "Hello world"
        assert state.current_text == "Hello world"

    @patch("mlx_qwen3_asr.streaming.init_streaming")
    @patch("mlx_qwen3_asr.streaming._ModelHolder")
    @patch("mlx_qwen3_asr.streaming.finish_streaming")
    def test_commit_returns_final_text(self, mock_finish, mock_holder, mock_init):
        config = ServerConfig()
        state = RealtimeSessionState(config)

        mock_model = Mock()
        mock_holder.get.return_value = (mock_model, None)
        mock_init.return_value = Mock(text="")
        mock_finish.return_value = Mock(text="Final transcript.")

        state.start()
        result = state.commit()

        assert result == "Final transcript."
        assert state.current_text == "Final transcript."

    @patch("mlx_qwen3_asr.streaming.init_streaming")
    @patch("mlx_qwen3_asr.streaming._ModelHolder")
    def test_clear(self, mock_holder, mock_init):
        config = ServerConfig()
        state = RealtimeSessionState(config)

        mock_model = Mock()
        mock_holder.get.return_value = (mock_model, None)
        mock_init.return_value = Mock(text="some text")

        state.start()
        state.new_response_id()
        state.clear()

        assert state.streaming_state is None
        assert state.current_text == ""
        assert state.response_id is None


class TestRealtimeTranscriber:
    @patch("mlx_qwen3_asr.streaming.feed_audio")
    def test_feed_audio(self, mock_feed):
        config = ServerConfig()
        model_obj = Mock()
        transcriber = RealtimeTranscriber(model_obj, config)

        audio = np.zeros(16000, dtype=np.float32)
        state = Mock()
        new_state = Mock()
        mock_feed.return_value = new_state

        result = transcriber.feed_audio(audio, state)

        mock_feed.assert_called_once_with(audio, state, model=model_obj)
        assert result == new_state

    @patch("mlx_qwen3_asr.streaming.finish_streaming")
    def test_finish(self, mock_finish):
        config = ServerConfig()
        model_obj = Mock()
        transcriber = RealtimeTranscriber(model_obj, config)

        state = Mock()
        final_state = Mock()
        mock_finish.return_value = final_state

        result = transcriber.finish(state)

        mock_finish.assert_called_once_with(state, model=model_obj)
        assert result == final_state


class TestRealtimeEventModels:
    def test_session_update_event(self):
        event = SessionUpdateEvent(
            session={"model": "whisper-1", "input_audio_format": "pcm16"}
        )
        assert event.type == "session.update"
        assert event.session.model == "whisper-1"

    def test_session_created_event(self):
        event = SessionCreatedEvent(
            session={"id": "sess_123", "model": "Qwen/Qwen3-ASR-1.7B"}
        )
        assert event.type == "session.created"
        assert event.session.id == "sess_123"

    def test_input_audio_buffer_append_event(self):
        audio_b64 = base64.b64encode(b"fake_audio").decode()
        event = InputAudioBufferAppendEvent(audio=audio_b64)
        assert event.type == "input_audio_buffer.append"
        assert event.audio == audio_b64

    def test_response_audio_transcript_delta_event(self):
        event = ResponseAudioTranscriptDeltaEvent(response_id="resp_123", delta="Hello")
        assert event.type == "response.audio_transcript.delta"
        assert event.response_id == "resp_123"
        assert event.delta == "Hello"

    def test_error_event(self):
        event = ErrorEvent(
            error={
                "type": "invalid_request_error",
                "message": "Bad request",
                "code": "invalid",
            }
        )
        assert event.type == "error"
        assert event.error.type == "invalid_request_error"


class TestWebSocketIntegration:
    @pytest.mark.asyncio
    async def test_websocket_session_created(self):
        from fastapi.testclient import TestClient
        from server.app import app

        with (
            patch("mlx_qwen3_asr.streaming._ModelHolder") as mock_holder,
            patch("mlx_qwen3_asr.streaming.init_streaming") as mock_init,
        ):
            mock_model = Mock()
            mock_holder.get.return_value = (mock_model, None)
            mock_init.return_value = Mock(text="")

            client = TestClient(app)

            with client.websocket_connect("/v1/realtime") as websocket:
                data = websocket.receive_json()
                assert data["type"] == "session.created"
                assert "session" in data
                assert data["session"]["id"].startswith("sess_")

    @pytest.mark.asyncio
    async def test_websocket_session_update(self):
        from fastapi.testclient import TestClient
        from server.app import app

        with (
            patch("mlx_qwen3_asr.streaming._ModelHolder") as mock_holder,
            patch("mlx_qwen3_asr.streaming.init_streaming") as mock_init,
        ):
            mock_model = Mock()
            mock_holder.get.return_value = (mock_model, None)
            mock_init.return_value = Mock(text="")

            client = TestClient(app)

            with client.websocket_connect("/v1/realtime") as websocket:
                websocket.receive_json()

                websocket.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "model": "whisper-1",
                            "input_audio_format": "pcm16",
                        },
                    }
                )

                response = websocket.receive_json()
                assert response["type"] == "session.updated"

    @pytest.mark.asyncio
    async def test_websocket_invalid_json(self):
        from fastapi.testclient import TestClient
        from server.app import app

        with (
            patch("mlx_qwen3_asr.streaming._ModelHolder") as mock_holder,
            patch("mlx_qwen3_asr.streaming.init_streaming") as mock_init,
        ):
            mock_model = Mock()
            mock_holder.get.return_value = (mock_model, None)
            mock_init.return_value = Mock(text="")

            client = TestClient(app)

            with client.websocket_connect("/v1/realtime") as websocket:
                websocket.receive_json()

                websocket.send_text("not valid json")

                response = websocket.receive_json()
                assert response["type"] == "error"
                assert response["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_websocket_unknown_event(self):
        from fastapi.testclient import TestClient
        from server.app import app

        with (
            patch("mlx_qwen3_asr.streaming._ModelHolder") as mock_holder,
            patch("mlx_qwen3_asr.streaming.init_streaming") as mock_init,
        ):
            mock_model = Mock()
            mock_holder.get.return_value = (mock_model, None)
            mock_init.return_value = Mock(text="")

            client = TestClient(app)

            with client.websocket_connect("/v1/realtime") as websocket:
                websocket.receive_json()

                websocket.send_json({"type": "unknown.event"})

                response = websocket.receive_json()
                assert response["type"] == "error"

    @pytest.mark.asyncio
    async def test_websocket_audio_commit(self):
        from fastapi.testclient import TestClient
        from server.app import app

        with (
            patch("mlx_qwen3_asr.streaming._ModelHolder") as mock_holder,
            patch("mlx_qwen3_asr.streaming.init_streaming") as mock_init,
            patch("mlx_qwen3_asr.streaming.finish_streaming") as mock_finish,
        ):
            mock_model = Mock()
            mock_holder.get.return_value = (mock_model, None)
            mock_init.return_value = Mock(text="")
            mock_finish.return_value = Mock(text="Transcribed text.")

            client = TestClient(app)

            with client.websocket_connect("/v1/realtime") as websocket:
                websocket.receive_json()

                websocket.send_json({"type": "input_audio_buffer.commit"})

                committed = websocket.receive_json()
                assert committed["type"] == "input_audio_buffer.committed"

                speech_started = websocket.receive_json()
                assert speech_started["type"] == "input_audio_buffer.speech_started"

                response_created = websocket.receive_json()
                assert response_created["type"] == "response.created"

                transcript_done = websocket.receive_json()
                assert transcript_done["type"] == "response.audio_transcript.done"

                speech_stopped = websocket.receive_json()
                assert speech_stopped["type"] == "input_audio_buffer.speech_stopped"

                response_done = websocket.receive_json()
                assert response_done["type"] == "response.done"

    @pytest.mark.asyncio
    async def test_websocket_audio_clear(self):
        from fastapi.testclient import TestClient
        from server.app import app

        with (
            patch("mlx_qwen3_asr.streaming._ModelHolder") as mock_holder,
            patch("mlx_qwen3_asr.streaming.init_streaming") as mock_init,
        ):
            mock_model = Mock()
            mock_holder.get.return_value = (mock_model, None)
            mock_init.return_value = Mock(text="")

            client = TestClient(app)

            with client.websocket_connect("/v1/realtime") as websocket:
                websocket.receive_json()

                websocket.send_json({"type": "input_audio_buffer.clear"})
