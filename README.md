# OpenAI-Compatible ASR Server

An OpenAI API compatible Automatic Speech Recognition (ASR) server powered by `mlx-qwen3-asr` for Apple Silicon (M1/M2/M3/M4).

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's Audio Transcriptions API
- **Multiple Output Formats**: JSON, text, SRT, VTT, and verbose JSON
- **SSE Streaming**: Real-time transcription updates via Server-Sent Events
- **Model Flexibility**: Use `whisper-1` alias or direct Qwen model IDs
- **Apple Silicon Optimized**: Leverages Metal GPU acceleration via MLX

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd openai-qwen-asr

# Install dependencies
uv sync
```

## Quick Start

```bash
# Start the server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The server will preload the ASR model on startup. Once you see "Model loaded successfully", the API is ready.

## API Usage

### Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

### Response Formats

```bash
# JSON (default)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"

# Plain text
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=text"

# SRT subtitles
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=srt"

# VTT subtitles
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=vtt"

# Verbose JSON (with timestamps)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json"
```

### Specify Language

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en"
```

### SSE Streaming Transcription

Get real-time transcription updates as the audio is processed:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Accept: text/event-stream" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "stream=true"
```

**SSE Event Format:**
```
event: transcript.partial
data: {"type":"transcript.partial","text":"Hello"}

event: transcript.partial
data: {"type":"transcript.partial","text":"Hello world"}

event: transcript.final
data: {"type":"transcript.final","text":"Hello world."}

data: [DONE]
```

**Python Example with Streaming:**

```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:8000/v1/audio/transcriptions",
    files={"file": open("audio.mp3", "rb")},
    data={"model": "whisper-1", "stream": "true"},
    timeout=60.0
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            print(data)
```

### List Available Models

```bash
curl http://localhost:8000/v1/models
```

### Health Check

```bash
curl http://localhost:8000/health
```

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # No authentication required
)

with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

print(transcript.text)
```

## Configuration

Configure the server using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | Default model ID |
| `DTYPE` | `fp16` | Data type (`fp16`, `bf16`, `fp32`) |
| `QUANTIZE_BITS` | (none) | Quantization bits (4 or 8) |
| `QUANTIZE_GROUP_SIZE` | `64` | Quantization group size |
| `MAX_FILE_SIZE_MB` | `100` | Max upload size (MB) |
| `MAX_CONCURRENT_REQUESTS` | `4` | Concurrent request limit |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `MAX_NEW_TOKENS` | `4096` | Max tokens to generate |
| `CHUNK_SIZE_SEC` | `2.0` | Streaming chunk size (seconds) |
| `MAX_CONTEXT_SEC` | `30.0` | Streaming max context (seconds) |

### Example

```bash
MODEL_ID=Qwen/Qwen3-ASR-0.6B QUANTIZE_BITS=4 uv run uvicorn server.app:app --port 8080
```

## Supported Audio Formats

- WAV, MP3, M4A, FLAC, OGG, WebM
- MP4, WebM (video containers)
- Maximum file size: 100MB (configurable)

## Model Mapping

| Alias | Model |
|-------|-------|
| `whisper-1` | `Qwen/Qwen3-ASR-1.7B` |
| `whisper` | `Qwen/Qwen3-ASR-1.7B` |
| `qwen-asr-0.6b` | `Qwen/Qwen3-ASR-0.6B` |
| `qwen-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B` |

You can also use direct Qwen model IDs like `Qwen/Qwen3-ASR-1.7B`.

## Project Structure

```
server/
├── app.py                 # FastAPI entry point
├── config.py              # Configuration
├── models.py              # Pydantic schemas
├── errors.py              # Error handling
├── asr/
│   ├── engine.py          # ASR engine wrapper
│   └── streaming.py       # SSE streaming transcriber
├── routes/
│   └── transcriptions.py  # Transcriptions endpoint
└── utils/
    ├── audio.py           # Audio utilities
    └── model_mapping.py   # Model name mapping
tests/
├── test_audio.py
├── test_errors.py
├── test_model_mapping.py
├── test_models.py
└── test_streaming.py
```

## Development

```bash
# Run tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_models.py
```

## Roadmap

- [x] Phase 1: Core Transcriptions API (non-streaming)
- [x] Phase 2: SSE Streaming Transcriptions
- [ ] Phase 3: Realtime WebSocket API
- [ ] Phase 4: Optimization & Concurrency Control
- [ ] Phase 5: Docker Deployment

## License

MIT