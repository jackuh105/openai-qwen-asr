#!/usr/bin/env python3
"""
Realtime WebSocket ASR client for testing.

Usage:
    uv run python test_realtime_client.py --file audio.mp3 --host localhost --port 8000

Requirements:
    - Server must be running: uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets


async def send_audio_chunks(
    ws: websockets.WebSocketClientProtocol,
    audio_path: str,
    chunk_size_sec: float = 1.0,
    sample_rate: int = 16000,
):
    """
    Read audio file and send as chunks over WebSocket.

    Args:
        ws: WebSocket connection
        audio_path: Path to audio file
        chunk_size_sec: Size of each chunk in seconds
        sample_rate: Target sample rate (16000 Hz)
    """
    audio, sr = sf.read(audio_path, dtype="float32")

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != sample_rate:
        import resampy

        audio = resampy.resample(audio, sr, sample_rate)

    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    pcm16 = (audio * 32767).astype(np.int16)

    chunk_samples = int(chunk_size_sec * sample_rate)
    total_samples = len(pcm16)

    print(f"\n🎵 Audio: {audio_path}")
    print(f"   Duration: {total_samples / sample_rate:.2f}s")
    print(f"   Chunks: {(total_samples + chunk_samples - 1) // chunk_samples}")
    print(f"   Chunk size: {chunk_size_sec}s\n")

    sent_chunks = 0
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = pcm16[start:end]

        audio_bytes = chunk.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode()

        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64,
                }
            )
        )

        sent_chunks += 1
        progress = end / total_samples * 100
        print(f"\r📤 Sending chunk {sent_chunks} ({progress:.0f}%)", end="", flush=True)

        await asyncio.sleep(0.1)

    print("\n")

    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    print("✅ Audio committed, waiting for transcription...\n")


async def receive_transcripts(ws: websockets.WebSocketClientProtocol):
    """Receive and print transcript events from WebSocket."""
    full_transcript = ""

    try:
        async for message in ws:
            event = json.loads(message)
            event_type = event.get("type", "")

            if event_type == "session.created":
                print(f"🔌 Session created: {event['session']['id']}")
                print(f"   Model: {event['session']['model']}\n")

            elif event_type == "session.updated":
                print(f"🔄 Session updated\n")

            elif event_type == "input_audio_buffer.committed":
                print("📥 Audio buffer committed\n")

            elif event_type == "input_audio_buffer.speech_started":
                print("🎤 Speech started\n")

            elif event_type == "response.created":
                print(f"📝 Response: {event['response_id']}")

            elif event_type == "response.audio_transcript.delta":
                delta = event["delta"]
                full_transcript += delta
                print(f"   Delta: {delta}")

            elif event_type == "response.audio_transcript.done":
                print(f"\n✨ Final transcript:\n   {event['transcript']}\n")

            elif event_type == "input_audio_buffer.speech_stopped":
                print("\n🔇 Speech stopped\n")

            elif event_type == "response.done":
                print(f"🏁 Response complete: {event['response_id']}\n")
                break

            elif event_type == "error":
                print(f"❌ Error: {event['error']['message']}")
                break

            else:
                print(f"   Event: {event_type}")

    except websockets.ConnectionClosed:
        print("\n🔌 Connection closed")

    return full_transcript


async def realtime_client(
    host: str, port: int, audio_path: str, chunk_size: float = 1.0
):
    """
    Connect to realtime ASR server and send audio for transcription.

    Args:
        host: Server host
        port: Server port
        audio_path: Path to audio file
        chunk_size: Chunk size in seconds
    """
    url = f"ws://{host}:{port}/v1/realtime"
    print(f"\n🌐 Connecting to {url}\n")

    try:
        async with websockets.connect(url) as ws:
            receive_task = asyncio.create_task(receive_transcripts(ws))

            await asyncio.sleep(0.5)

            await send_audio_chunks(ws, audio_path, chunk_size_sec=chunk_size)

            await receive_task

    except ConnectionRefusedError:
        print(f"❌ Could not connect to {url}")
        print("   Make sure the server is running:")
        print(f"   uv run uvicorn server.app:app --host {host} --port {port}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test realtime ASR WebSocket API")
    parser.add_argument("--file", "-f", required=True, help="Path to audio file")
    parser.add_argument(
        "--host", default="localhost", help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=1.0,
        help="Chunk size in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    asyncio.run(
        realtime_client(
            host=args.host,
            port=args.port,
            audio_path=str(audio_path),
            chunk_size=args.chunk_size,
        )
    )


if __name__ == "__main__":
    main()
