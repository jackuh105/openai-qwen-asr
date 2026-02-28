import pytest
from server.utils.model_mapping import resolve_model, MODEL_MAPPING


class TestResolveModel:
    def test_whisper_1_mapping(self):
        assert resolve_model("whisper-1") == "Qwen/Qwen3-ASR-1.7B"

    def test_whisper_mapping(self):
        assert resolve_model("whisper") == "Qwen/Qwen3-ASR-1.7B"

    def test_qwen_asr_06b_mapping(self):
        assert resolve_model("qwen-asr-0.6b") == "Qwen/Qwen3-ASR-0.6B"

    def test_qwen_asr_17b_mapping(self):
        assert resolve_model("qwen-asr-1.7b") == "Qwen/Qwen3-ASR-1.7B"

    def test_case_insensitive(self):
        assert resolve_model("WHISPER-1") == "Qwen/Qwen3-ASR-1.7B"
        assert resolve_model("WhIsPeR") == "Qwen/Qwen3-ASR-1.7B"

    def test_direct_qwen_id_passthrough(self):
        assert resolve_model("Qwen/Qwen3-ASR-0.6B") == "Qwen/Qwen3-ASR-0.6B"
        assert resolve_model("Qwen/Qwen3-ASR-1.7B") == "Qwen/Qwen3-ASR-1.7B"

    def test_unknown_model_passthrough(self):
        assert resolve_model("some-unknown-model") == "some-unknown-model"
        assert resolve_model("custom-model-v2") == "custom-model-v2"


class TestModelMapping:
    def test_mapping_dict_exists(self):
        assert isinstance(MODEL_MAPPING, dict)
        assert "whisper-1" in MODEL_MAPPING
        assert "whisper" in MODEL_MAPPING
