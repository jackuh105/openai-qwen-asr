import tempfile
import os
from typing import Tuple
import numpy as np

from server.errors import invalid_file_error


def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Load audio from bytes and convert to 16kHz mono float32.

    Uses mlx_qwen3_asr.load_audio internally which handles various formats
    via ffmpeg.

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is float32 mono
    """
    from mlx_qwen3_asr import load_audio

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio = load_audio(tmp_path)
        return audio, 16000
    finally:
        os.unlink(tmp_path)


def load_audio_from_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from file path and convert to 16kHz mono float32.

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is float32 mono
    """
    from mlx_qwen3_asr import load_audio

    audio = load_audio(file_path)
    return audio, 16000


def get_audio_duration(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Calculate audio duration in seconds.
    """
    return len(audio) / sample_rate


def format_timestamp(seconds: float) -> str:
    """
    Format timestamp for SRT format: HH:MM:SS,mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    Format timestamp for VTT format: HH:MM:SS.mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def segments_to_srt(segments: list) -> str:
    """
    Convert segments with timestamps to SRT format.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys

    Returns:
        SRT formatted string
    """
    lines = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def segments_to_vtt(segments: list) -> str:
    """
    Convert segments with timestamps to VTT format.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys

    Returns:
        VTT formatted string
    """
    lines = ["WEBVTT", ""]
    for segment in segments:
        start = format_timestamp_vtt(segment["start"])
        end = format_timestamp_vtt(segment["end"])
        text = segment["text"].strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)
