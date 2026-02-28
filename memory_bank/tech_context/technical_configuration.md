# Technical Configuration

## Platform Requirements
- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4)
- **OS**: macOS with Metal support
- **Python**: 3.10+

## Dependencies

### Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.109.0 | Web framework |
| uvicorn | >=0.27.0 | ASGI server |
| mlx-qwen3-asr | >=0.1.0 | ASR engine |
| python-multipart | >=0.0.6 | Multipart form handling |
| websockets | >=12.0 | WebSocket support |
| pydantic | >=2.0.0 | Data validation |

### Development Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=8.0.0 | Testing |
| httpx | >=0.26.0 | HTTP client for tests |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | Default model ID |
| `DTYPE` | `fp16` | Data type |
| `QUANTIZE_BITS` | (none) | Quantization bits (4 or 8) |
| `QUANTIZE_GROUP_SIZE` | `64` | Quantization group size |
| `MAX_FILE_SIZE_MB` | `100` | Max upload size (MB) |
| `MAX_CONCURRENT_REQUESTS` | `4` | Concurrent request limit |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `CHUNK_SIZE_SEC` | `2.0` | Streaming chunk size (seconds) |
| `MAX_CONTEXT_SEC` | `30.0` | Max streaming context (seconds) |
| `MAX_NEW_TOKENS` | `4096` | Max tokens to generate |

## Audio Format Requirements

### Transcriptions API
- **Supported formats**: wav, mp3, m4a, flac, mp4, etc. (via ffmpeg)
- **Internal format**: 16kHz mono float32
- **Max file size**: Configurable (default 100MB)

### Realtime API
- **Format**: PCM16 (16-bit signed integer, little-endian)
- **Sample rate**: 16000 Hz
- **Channels**: Mono
- **Encoding**: Base64

## Memory Requirements
- **0.6B model**: ~1.2 GB (fp16)
- **1.7B model**: ~3.4 GB (fp16)
- Quantization reduces memory usage proportionally