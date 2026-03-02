import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.config import config
from server.models import (
    RealtimeSessionConfig,
    RealtimeSession,
    SessionUpdateEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ResponseCreatedEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseDoneEvent,
    ErrorEvent,
    ErrorDetail,
)
from server.asr.realtime import RealtimeSessionState
from server.utils.model_mapping import resolve_model


router = APIRouter(tags=["Realtime"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str) -> None:
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_json(self, session_id: str, data: dict) -> None:
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)


manager = ConnectionManager()


@router.websocket("/v1/realtime")
async def websocket_endpoint(websocket: WebSocket):
    session_state = RealtimeSessionState(config)
    session_state.start()

    await manager.connect(websocket, session_state.session_id)

    session_created = SessionCreatedEvent(
        session=RealtimeSession(
            id=session_state.session_id,
            model=config.model_id,
        )
    )
    await websocket.send_json(session_created.model_dump())

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                event_type = message.get("type")

                if event_type == "session.update":
                    await _handle_session_update(websocket, session_state, message)

                elif event_type == "input_audio_buffer.append":
                    await _handle_audio_append(websocket, session_state, message)

                elif event_type == "input_audio_buffer.commit":
                    await _handle_audio_commit(websocket, session_state)

                elif event_type == "input_audio_buffer.clear":
                    await _handle_audio_clear(websocket, session_state)

                else:
                    await _send_error(
                        websocket,
                        f"Unknown event type: {event_type}",
                        "invalid_request_error",
                    )

            except json.JSONDecodeError:
                await _send_error(websocket, "Invalid JSON", "invalid_request_error")

    except WebSocketDisconnect:
        manager.disconnect(session_state.session_id)
    except Exception as e:
        await _send_error(websocket, str(e), "server_error")
        manager.disconnect(session_state.session_id)


async def _handle_session_update(
    websocket: WebSocket,
    session_state: RealtimeSessionState,
    message: dict,
) -> None:
    try:
        event = SessionUpdateEvent(**message)
    except Exception:
        await _send_error(
            websocket, "Invalid session.update message", "invalid_request_error"
        )
        return

    session_data = event.session
    resolved_model = resolve_model(session_data.model)
    session_state.model = resolved_model

    if session_data.input_audio_format != "pcm16":
        await _send_error(
            websocket,
            f"Unsupported audio format: {session_data.input_audio_format}. Only pcm16 is supported.",
            "invalid_request_error",
        )
        return

    session_updated = SessionUpdatedEvent(
        session=RealtimeSession(
            id=session_state.session_id,
            model=session_state.model,
        )
    )
    await websocket.send_json(session_updated.model_dump())


async def _handle_audio_append(
    websocket: WebSocket,
    session_state: RealtimeSessionState,
    message: dict,
) -> None:
    try:
        event = InputAudioBufferAppendEvent(**message)
    except Exception:
        await _send_error(
            websocket,
            "Invalid input_audio_buffer.append message",
            "invalid_request_error",
        )
        return

    try:
        delta = session_state.append_audio(event.audio)

        if delta:
            if session_state.response_id is None:
                session_state.new_response_id()

            resp_id = session_state.response_id
            assert resp_id is not None

            response_created = ResponseCreatedEvent(response_id=resp_id)
            await websocket.send_json(response_created.model_dump())

            delta_event = ResponseAudioTranscriptDeltaEvent(
                response_id=resp_id,
                delta=delta,
            )
            await websocket.send_json(delta_event.model_dump())

    except Exception as e:
        await _send_error(
            websocket, f"Failed to process audio: {str(e)}", "transcription_failed"
        )


async def _handle_audio_commit(
    websocket: WebSocket,
    session_state: RealtimeSessionState,
) -> None:
    try:
        committed = InputAudioBufferCommittedEvent()
        await websocket.send_json(committed.model_dump())

        speech_started = InputAudioBufferSpeechStartedEvent()
        await websocket.send_json(speech_started.model_dump())

        final_text = session_state.commit()

        if session_state.response_id is None:
            session_state.new_response_id()

        resp_id = session_state.response_id
        assert resp_id is not None

        response_created = ResponseCreatedEvent(response_id=resp_id)
        await websocket.send_json(response_created.model_dump())

        done_event = ResponseAudioTranscriptDoneEvent(
            response_id=resp_id,
            transcript=final_text,
        )
        await websocket.send_json(done_event.model_dump())

        speech_stopped = InputAudioBufferSpeechStoppedEvent()
        await websocket.send_json(speech_stopped.model_dump())

        response_done = ResponseDoneEvent(response_id=resp_id)
        await websocket.send_json(response_done.model_dump())

        session_state.clear()

    except Exception as e:
        await _send_error(
            websocket, f"Failed to commit audio: {str(e)}", "transcription_failed"
        )


async def _handle_audio_clear(
    websocket: WebSocket,
    session_state: RealtimeSessionState,
) -> None:
    session_state.clear()


async def _send_error(
    websocket: WebSocket,
    message: str,
    error_type: str,
    code: Optional[str] = None,
) -> None:
    error_event = ErrorEvent(
        error=ErrorDetail(
            type=error_type,
            message=message,
            code=code,
        )
    )
    await websocket.send_json(error_event.model_dump())
