import pytest
from server.errors import (
    OpenAIError,
    ErrorDetail,
    ErrorResponse,
    invalid_file_error,
    file_too_large_error,
    invalid_model_error,
    invalid_response_format_error,
    transcription_failed_error,
    server_busy_error,
)


class TestOpenAIError:
    def test_basic_error(self):
        error = OpenAIError("Test error message")
        assert error.message == "Test error message"
        assert error.error_type == "invalid_request_error"
        assert error.status_code == 400

    def test_error_with_all_params(self):
        error = OpenAIError(
            message="Custom error",
            error_type="server_error",
            param="file",
            code="invalid_file",
            status_code=500,
        )
        assert error.message == "Custom error"
        assert error.error_type == "server_error"
        assert error.param == "file"
        assert error.code == "invalid_file"
        assert error.status_code == 500

    def test_to_response(self):
        error = OpenAIError(
            message="Test error",
            error_type="invalid_request_error",
            param="model",
            code="invalid_model",
        )
        response = error.to_response()
        assert "error" in response
        assert response["error"]["message"] == "Test error"
        assert response["error"]["type"] == "invalid_request_error"
        assert response["error"]["param"] == "model"
        assert response["error"]["code"] == "invalid_model"


class TestErrorFactories:
    def test_invalid_file_error(self):
        error = invalid_file_error("Unsupported format")
        assert error.message == "Unsupported format"
        assert error.code == "invalid_file"
        assert error.param == "file"
        assert error.status_code == 400

    def test_file_too_large_error(self):
        error = file_too_large_error(100)
        assert "100MB" in error.message
        assert error.code == "file_too_large"
        assert error.status_code == 400

    def test_invalid_model_error(self):
        error = invalid_model_error("bad-model")
        assert "bad-model" in error.message
        assert error.code == "invalid_model"
        assert error.param == "model"

    def test_invalid_response_format_error(self):
        error = invalid_response_format_error("bad_format")
        assert "bad_format" in error.message
        assert error.code == "invalid_response_format"

    def test_transcription_failed_error(self):
        error = transcription_failed_error("Model crashed")
        assert error.message == "Model crashed"
        assert error.code == "transcription_failed"
        assert error.status_code == 500

    def test_server_busy_error(self):
        error = server_busy_error()
        assert "busy" in error.message.lower()
        assert error.code == "server_busy"
        assert error.status_code == 503


class TestErrorModels:
    def test_error_detail_model(self):
        detail = ErrorDetail(
            message="Test",
            type="invalid_request_error",
            param="file",
            code="invalid_file",
        )
        assert detail.message == "Test"
        assert detail.type == "invalid_request_error"

    def test_error_response_model(self):
        detail = ErrorDetail(message="Test error")
        response = ErrorResponse(error=detail)
        assert response.error.message == "Test error"
