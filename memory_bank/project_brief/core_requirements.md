# Core Requirements

## Objective
Build an OpenAI API compatible ASR (Automatic Speech Recognition) server using `mlx-qwen3-asr` as the ASR engine.

## APIs to Implement

### 1. Audio Transcriptions API
- **Endpoint**: `POST /v1/audio/transcriptions`
- **Non-streaming mode**: Client uploads complete audio, server returns complete transcription
- **SSE streaming mode**: Client uploads complete audio, server streams text deltas via Server-Sent Events

### 2. Realtime API
- **Endpoint**: `wss://<host>/v1/realtime`
- **Protocol**: WebSocket bidirectional communication
- **Scope**: ASR events only (no TTS, no function calling)

## Key Features
- OpenAI API compatibility for easy integration with existing OpenAI clients
- Support for multiple response formats (json, text, srt, vtt, verbose_json)
- Streaming transcription with real-time text deltas
- WebSocket-based realtime transcription for live audio input

## Technology Stack
- **Framework**: FastAPI
- **ASR Engine**: mlx-qwen3-asr
- **Platform**: Apple Silicon (M1/M2/M3/M4)