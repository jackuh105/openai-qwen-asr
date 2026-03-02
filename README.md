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
- [x] Phase 3: Realtime WebSocket API
- [ ] Phase 4: Optimization & Concurrency Control
- [ ] Phase 5: Docker Deployment

## Realtime WebSocket API

Connect to `ws://localhost:8000/v1/realtime` for bidirectional real-time transcription.

### Audio Format

- **Format**: PCM16 (16-bit signed integer, little-endian)
- **Sample Rate**: 16000 Hz
- **Channels**: Mono
- **Encoding**: Base64

### Event Types

**Client → Server:**

| Event | Description |
|-------|-------------|
| `session.update` | Update session settings (model, language) |
| `input_audio_buffer.append` | Append audio data (base64 PCM16) |
| `input_audio_buffer.commit` | Commit buffer and trigger transcription |
| `input_audio_buffer.clear` | Clear audio buffer |

**Server → Client:**

| Event | Description |
|-------|-------------|
| `session.created` | Session established |
| `session.updated` | Session settings updated |
| `input_audio_buffer.committed` | Audio buffer committed |
| `input_audio_buffer.speech_started` | Speech detected |
| `input_audio_buffer.speech_stopped` | Speech ended |
| `response.created` | Response generation started |
| `response.audio_transcript.delta` | Incremental text |
| `response.audio_transcript.done` | Final transcript |
| `response.done` | Response complete |
| `error` | Error event |

### Example Flow

```
Client                                Server
  |                                     |
  |<-------- session.created ----------|
  |                                     |
  |------ session.update ------------->|
  |<-------- session.updated ----------|
  |                                     |
  |--- input_audio_buffer.append ----->|  (base64 PCM16)
  |--- input_audio_buffer.append ----->|  (multiple times)
  |--- input_audio_buffer.commit ----->|
  |                                     |
  |<---- response.created -------------|
  |<---- response.audio_transcript.delta --|
  |<---- response.audio_transcript.done ---|
  |<---- response.done ----------------|
```

### Python Example

```python
import asyncio
import websockets
import json
import base64

async def realtime_transcribe():
    async with websockets.connect("ws://localhost:8000/v1/realtime") as ws:
        # Receive session.created
        msg = json.loads(await ws.recv())
        print(f"Session: {msg['session']['id']}")

        # Send audio (example: generate dummy PCM16)
        audio_data = b'\x00\x00' * 16000  # 1 second of silence
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_data).decode()
        }))

        # Commit and receive transcription
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async for msg in ws:
            event = json.loads(msg)
            print(f"Event: {event['type']}")
            if event["type"] == "response.audio_transcript.done":
                print(f"Transcript: {event['transcript']}")
                break

asyncio.run(realtime_transcribe())
```

### Test Client

A test client script is provided for testing the realtime API with audio files:

```bash
# Terminal 1: Start the server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Run the realtime client
uv run python test_realtime_client.py --file audio.mp3 --host localhost --port 8000
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--file, -f` | (required) | Path to audio file |
| `--host` | `localhost` | Server host |
| `--port` | `8000` | Server port |
| `--chunk-size` | `1.0` | Chunk size in seconds |

**What it does:**
1. Connects to the WebSocket endpoint at `/v1/realtime`
2. Loads and resamples audio to 16kHz mono
3. Sends audio chunks as base64-encoded PCM16
4. Prints all events including text deltas and final transcript

## License

MIT