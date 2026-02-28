# Technical Restrictions

## Platform Restrictions

### Apple Silicon Only
- mlx-qwen3-asr requires Apple Silicon (M1/M2/M3/M4)
- Metal GPU acceleration is mandatory
- Will not work on Intel Macs or other platforms

### Python Version
- Minimum Python 3.10 required by mlx-qwen3-asr

## Feature Restrictions

### ASR Only
- No Text-to-Speech (TTS) capability
- No function calling support
- Realtime API limited to transcription events

### Language Support
- Supports 30 core languages + 22 Chinese dialects
- Language detection is automatic when not specified
- Some languages may have varying quality

### Model Limitations
- Cannot use temperature parameter (not supported by engine)
- Cannot use prompt parameter (not supported by engine)
- No custom vocabulary injection

## API Restrictions

### Response Formats
- `json`, `text`, `srt`, `vtt`, `verbose_json` supported
- Other formats return 400 error

### File Upload
- Maximum file size configurable (default 100MB)
- Requires ffmpeg for non-WAV formats

### Concurrency
- Limited by available memory
- Default limit: 4 concurrent requests
- Each session holds GPU memory

## Streaming Restrictions

### SSE Streaming
- Client must upload complete file first
- Server-side chunking for streaming output

### Realtime API
- No turn detection/VAD (planned for future)
- Audio must be pre-chunked by client
- Base64 encoding required (overhead)

## Performance Considerations

### Latency
- 0.6B: ~0.08x real-time factor
- 1.7B: ~0.27x real-time factor
- Quantization improves speed but may affect quality

### Memory
- Model loaded once and cached
- Each streaming session uses additional memory
- Memory usage scales with max_context_sec