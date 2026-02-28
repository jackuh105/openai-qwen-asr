MODEL_MAPPING = {
    "whisper-1": "Qwen/Qwen3-ASR-1.7B",
    "whisper": "Qwen/Qwen3-ASR-1.7B",
    "qwen-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
}


def resolve_model(model_param: str) -> str:
    """
    Resolve model parameter to Qwen model ID.

    1. Check if it's a mapped alias (e.g., 'whisper-1')
    2. Otherwise, assume it's a direct Qwen ID
    """
    return MODEL_MAPPING.get(model_param.lower(), model_param)
