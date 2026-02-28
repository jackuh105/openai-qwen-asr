# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         OpenAI Clients                          │
│              (Python SDK, JavaScript, HTTP clients)             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Server                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ /v1/audio/      │  │ /v1/realtime    │  │ Error           │ │
│  │ transcriptions  │  │ (WebSocket)     │  │ Handling        │ │
│  └────────┬────────┘  └────────┬────────┘  └─────────────────┘ │
│           │                    │                                 │
│           └────────┬───────────┘                                 │
│                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    ASR Engine Layer                         ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ ││
│  │  │ Non-stream  │  │ SSE Stream  │  │ Realtime Session    │ ││
│  │  │ Transcribe  │  │ Transcribe  │  │ (WebSocket handler) │ ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    mlx-qwen3-asr Library                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Session         │  │ Streaming API   │  │ load_audio()    │ │
│  │ transcribe()    │  │ init/feed/finish│  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Routes Layer (`server/routes/`)
- HTTP endpoint handlers
- Request parsing and validation
- Response formatting

### ASR Layer (`server/asr/`)
- `engine.py`: Non-streaming transcription wrapper
- `streaming.py`: SSE streaming state management
- `realtime.py`: WebSocket session management

### Utils Layer (`server/utils/`)
- Audio loading and format conversion
- Model name mapping (OpenAI → Qwen)