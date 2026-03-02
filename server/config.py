from dataclasses import dataclass
from typing import Optional, Any
import os


DTYPE_MAP = {
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}


@dataclass
class ServerConfig:
    model_id: str = "Qwen/Qwen3-ASR-1.7B"
    dtype: str = "float16"
    quantize_bits: Optional[int] = None
    quantize_group_size: int = 64

    sample_rate: int = 16000
    max_file_size_mb: int = 100

    chunk_size_sec: float = 2.0
    max_context_sec: float = 30.0

    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 4

    max_new_tokens: int = 4096

    def get_mlx_dtype(self) -> Any:
        import mlx.core as mx

        dtype_name = DTYPE_MAP.get(self.dtype, self.dtype)
        return getattr(mx, dtype_name, mx.float16)

    @classmethod
    def from_env(cls) -> "ServerConfig":
        quantize_bits_raw = os.getenv("QUANTIZE_BITS")
        quantize_bits = int(quantize_bits_raw) if quantize_bits_raw else None

        dtype_str = os.getenv("DTYPE", "float16")
        dtype_str = DTYPE_MAP.get(dtype_str, dtype_str)

        return cls(
            model_id=os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B"),
            dtype=dtype_str,
            quantize_bits=quantize_bits,
            quantize_group_size=int(os.getenv("QUANTIZE_GROUP_SIZE", "64")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            chunk_size_sec=float(os.getenv("CHUNK_SIZE_SEC", "2.0")),
            max_context_sec=float(os.getenv("MAX_CONTEXT_SEC", "30.0")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "4096")),
        )


config = ServerConfig.from_env()
