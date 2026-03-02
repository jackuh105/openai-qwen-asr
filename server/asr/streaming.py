from typing import AsyncIterator, Optional, TYPE_CHECKING
import numpy as np
import mlx.core as mx

from server.config import ServerConfig

if TYPE_CHECKING:
    from mlx_qwen3_asr.streaming import StreamingState


class StreamingTranscriber:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._model_obj = None

    def init_state(self, language: Optional[str] = None) -> "StreamingState":
        from mlx_qwen3_asr.streaming import init_streaming, _ModelHolder  # type: ignore

        dtype = self.config.get_mlx_dtype()

        model_obj, _ = _ModelHolder.get(self.config.model_id, dtype=dtype)  # type: ignore
        self._model_obj = model_obj

        return init_streaming(
            model=self.config.model_id,
            chunk_size_sec=self.config.chunk_size_sec,
            max_context_sec=self.config.max_context_sec,
            sample_rate=self.config.sample_rate,
            dtype=dtype,
            max_new_tokens=self.config.max_new_tokens,
            language=language,
        )

    def feed_audio(
        self, audio: np.ndarray, state: "StreamingState"
    ) -> "StreamingState":
        from mlx_qwen3_asr.streaming import feed_audio

        return feed_audio(audio, state, model=self._model_obj)

    def finish(self, state: "StreamingState") -> "StreamingState":
        from mlx_qwen3_asr.streaming import finish_streaming

        return finish_streaming(state, model=self._model_obj)

    async def transcribe_stream(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> AsyncIterator[str]:
        import asyncio

        state = self.init_state(language=language)

        chunk_samples = int(self.config.chunk_size_sec * self.config.sample_rate)
        total_samples = len(audio)

        prev_text = ""
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]

            state = self.feed_audio(chunk, state)

            if state.text != prev_text:
                yield state.text
                prev_text = state.text

            await asyncio.sleep(0)

        state = self.finish(state)
        if state.text != prev_text:
            yield state.text

    async def transcribe_stream_with_deltas(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> AsyncIterator[tuple[str, str]]:
        import asyncio

        state = self.init_state(language=language)

        chunk_samples = int(self.config.chunk_size_sec * self.config.sample_rate)
        total_samples = len(audio)

        prev_text = ""
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]

            state = self.feed_audio(chunk, state)

            if state.text != prev_text:
                yield ("partial", state.text)
                prev_text = state.text

            await asyncio.sleep(0)

        state = self.finish(state)
        if state.text != prev_text:
            yield ("partial", state.text)

        yield ("final", state.text)
