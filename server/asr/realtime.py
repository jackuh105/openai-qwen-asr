import base64
from typing import Optional, TYPE_CHECKING
import uuid
import numpy as np

from server.config import ServerConfig

if TYPE_CHECKING:
    from mlx_qwen3_asr.streaming import StreamingState


class RealtimeSessionState:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.session_id = f"sess_{uuid.uuid4().hex[:12]}"
        self.response_id: Optional[str] = None
        self.model: str = config.model_id
        self.language: Optional[str] = None
        self.streaming_state: Optional["StreamingState"] = None
        self.current_text: str = ""
        self._transcriber: Optional["RealtimeTranscriber"] = None

    def start(self) -> None:
        from mlx_qwen3_asr.streaming import init_streaming, _ModelHolder

        dtype = self.config.get_mlx_dtype()
        model_obj, _ = _ModelHolder.get(self.config.model_id, dtype=dtype)

        self._transcriber = RealtimeTranscriber(
            model_obj=model_obj,
            config=self.config,
        )
        self.streaming_state = init_streaming(
            model=self.config.model_id,
            chunk_size_sec=self.config.chunk_size_sec,
            max_context_sec=self.config.max_context_sec,
            sample_rate=self.config.sample_rate,
            dtype=dtype,
            max_new_tokens=self.config.max_new_tokens,
            language=self.language,
        )

    def append_audio(self, base64_audio: str) -> Optional[str]:
        if self.streaming_state is None or self._transcriber is None:
            raise RuntimeError("Session not started. Call start() first.")

        pcm_bytes = base64.b64decode(base64_audio)
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0

        self.streaming_state = self._transcriber.feed_audio(
            pcm_float, self.streaming_state
        )

        if self.streaming_state.text != self.current_text:
            delta = self.streaming_state.text[len(self.current_text) :]
            self.current_text = self.streaming_state.text
            return delta
        return None

    def commit(self) -> str:
        if self.streaming_state is None or self._transcriber is None:
            raise RuntimeError("Session not started. Call start() first.")

        self.streaming_state = self._transcriber.finish(self.streaming_state)
        self.current_text = self.streaming_state.text
        return self.current_text

    def clear(self) -> None:
        self.streaming_state = None
        self.current_text = ""
        self.response_id = None

    def new_response_id(self) -> str:
        self.response_id = f"resp_{uuid.uuid4().hex[:12]}"
        return self.response_id


class RealtimeTranscriber:
    def __init__(self, model_obj, config: ServerConfig):
        self._model_obj = model_obj
        self.config = config

    def feed_audio(
        self, audio: np.ndarray, state: "StreamingState"
    ) -> "StreamingState":
        from mlx_qwen3_asr.streaming import feed_audio

        return feed_audio(audio, state, model=self._model_obj)

    def finish(self, state: "StreamingState") -> "StreamingState":
        from mlx_qwen3_asr.streaming import finish_streaming

        return finish_streaming(state, model=self._model_obj)
