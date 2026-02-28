from typing import Optional, Any, Dict, List
import numpy as np

from server.config import ServerConfig


class ASREngine:
    _instance: Optional["ASREngine"] = None
    _session: Any = None
    _config: Optional[ServerConfig] = None

    def __new__(cls, config: Optional[ServerConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is not None:
            ASREngine._config = config

    @classmethod
    def get_instance(cls, config: Optional[ServerConfig] = None) -> "ASREngine":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def load_model(cls) -> None:
        """
        Load the ASR model. Should be called once at startup.
        """
        if cls._session is not None:
            return

        from mlx_qwen3_asr import Session

        config = cls._config or ServerConfig()

        kwargs: Dict[str, Any] = {
            "model": config.model_id,
            "dtype": config.dtype,
        }

        if config.quantize_bits is not None:
            kwargs["quantize_bits"] = config.quantize_bits
            kwargs["quantize_group_size"] = config.quantize_group_size

        cls._session = Session(**kwargs)

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._session is not None

    @classmethod
    def transcribe(
        cls,
        audio: np.ndarray,
        language: Optional[str] = None,
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio array (16kHz mono float32)
            language: Language code (e.g., 'en', 'zh')
            return_timestamps: Whether to return word-level timestamps

        Returns:
            Dict with 'text' and optionally 'segments' or 'words'
        """
        if cls._session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        result = cls._session.transcribe(
            audio, language=language, return_timestamps=return_timestamps
        )

        return result

    @classmethod
    def transcribe_file(
        cls,
        file_path: str,
        language: Optional[str] = None,
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            file_path: Path to audio file
            language: Language code (e.g., 'en', 'zh')
            return_timestamps: Whether to return word-level timestamps

        Returns:
            Dict with 'text' and optionally 'segments' or 'words'
        """
        if cls._session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        result = cls._session.transcribe(
            file_path, language=language, return_timestamps=return_timestamps
        )

        return result

    @classmethod
    def detect_language(cls, audio: np.ndarray) -> Optional[str]:
        """
        Detect language from audio.

        Args:
            audio: Audio array (16kHz mono float32)

        Returns:
            Detected language code or None
        """
        if cls._session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return cls._session.detect_language(audio)
