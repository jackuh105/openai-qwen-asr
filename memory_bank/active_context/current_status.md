# Current Project Status

## Status: Phase 1 Complete ✅, Ready for Phase 2

## Session: 2026-03-02

### Completed (Phase 1)
- [x] Created project directory structure (`server/`, `tests/`, `server/asr/`, `server/routes/`, `server/utils/`)
- [x] Implemented `server/config.py` - Server configuration with environment variables and dtype mapping
- [x] Implemented `server/utils/model_mapping.py` - OpenAI model name to Qwen ID mapping
- [x] Implemented `pyproject.toml` - Project config with uv
- [x] Implemented `server/models.py` - Pydantic request/response schemas
- [x] Implemented `server/errors.py` - OpenAI-compatible error format
- [x] Implemented `server/utils/audio.py` - Audio loading, SRT/VTT formatting
- [x] Implemented `server/asr/engine.py` - ASR engine wrapper (singleton pattern)
- [x] Implemented `server/routes/transcriptions.py` - POST /v1/audio/transcriptions endpoint
- [x] Implemented `server/app.py` - FastAPI entry point with model preloading
- [x] Unit tests - 45 tests passing

### Files Created
```
server/
├── __init__.py
├── app.py                   ✅ DONE
├── config.py                ✅ DONE
├── errors.py                ✅ DONE
├── models.py                ✅ DONE
├── asr/
│   ├── __init__.py
│   └── engine.py            ✅ DONE
├── routes/
│   ├── __init__.py
│   └── transcriptions.py    ✅ DONE
└── utils/
    ├── __init__.py
    ├── audio.py             ✅ DONE
    └── model_mapping.py     ✅ DONE
tests/
├── __init__.py
├── test_audio.py            ✅ DONE
├── test_errors.py           ✅ DONE
├── test_model_mapping.py    ✅ DONE
└── test_models.py           ✅ DONE
pyproject.toml               ✅ DONE
```

### Key Discoveries
1. **mlx-qwen3-asr dtype issue**: `Session` expects `mx.Dtype` objects (like `mx.float16`), NOT strings. Fixed with `DTYPE_MAP` in config.
2. **Pydantic V2 config**: Use `model_config = ConfigDict(extra="ignore")` for ignoring extra fields.
3. **TranscriptionResult is dataclass**: Use attribute access (`result.text`) instead of dict methods.

### Next Steps
1. Phase 2: SSE streaming for transcriptions
2. Phase 3: Realtime WebSocket API
3. Integration tests with actual ASR model
4. Docker deployment

## Design Decisions Confirmed

| Decision | Choice |
|----------|--------|
| Model parameter | Map `whisper-1` → `Qwen/Qwen3-ASR-1.7B`, accept direct Qwen IDs |
| Unsupported params | Silently ignore `temperature`, `prompt` |
| Timestamps | Auto-enable for `srt`/`vtt` formats |
| Default model | `Qwen/Qwen3-ASR-1.7B` |
| JSON response | Strict OpenAI: `{"text": "..."}` |
| Language format | Pass through to mlx-qwen3-asr |
| Realtime audio | base64 PCM16 |
| Authentication | None |
| Quantization | Configurable via env var |
| ASR Engine | Singleton pattern with lazy loading |
| Dtype handling | `DTYPE_MAP` converts string config to `mx.Dtype` |