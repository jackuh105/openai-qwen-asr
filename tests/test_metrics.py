import time
import pytest

from server.metrics import ServerMetrics, RequestMetrics, RealtimeSessionMetrics


class TestRequestMetrics:
    def test_init(self):
        metrics = RequestMetrics()
        assert metrics.count == 0
        assert metrics.total_time_ms == 0.0
        assert metrics.min_time_ms == float("inf")
        assert metrics.max_time_ms == 0.0
        assert metrics.errors == 0

    def test_avg_time_empty(self):
        metrics = RequestMetrics()
        assert metrics.avg_time_ms == 0.0

    def test_avg_time_with_values(self):
        metrics = RequestMetrics()
        metrics.count = 3
        metrics.total_time_ms = 300.0
        assert metrics.avg_time_ms == 100.0


class TestRealtimeSessionMetrics:
    def test_init(self):
        metrics = RealtimeSessionMetrics()
        assert metrics.active_sessions == 0
        assert metrics.total_sessions == 0
        assert metrics.audio_bytes_received == 0
        assert metrics.transcription_events == 0


class TestServerMetrics:
    def test_init(self):
        metrics = ServerMetrics()
        assert metrics.start_time > 0
        assert metrics.requests == {}
        assert metrics.realtime.active_sessions == 0
        assert len(metrics.recent_latencies) == 0

    def test_record_request(self):
        metrics = ServerMetrics()
        metrics.record_request("/v1/audio/transcriptions", 100.0)

        assert "/v1/audio/transcriptions" in metrics.requests
        req = metrics.requests["/v1/audio/transcriptions"]
        assert req.count == 1
        assert req.total_time_ms == 100.0
        assert req.min_time_ms == 100.0
        assert req.max_time_ms == 100.0
        assert req.errors == 0

    def test_record_request_multiple(self):
        metrics = ServerMetrics()
        metrics.record_request("/v1/audio/transcriptions", 100.0)
        metrics.record_request("/v1/audio/transcriptions", 200.0)
        metrics.record_request("/v1/audio/transcriptions", 50.0)

        req = metrics.requests["/v1/audio/transcriptions"]
        assert req.count == 3
        assert req.total_time_ms == 350.0
        assert req.min_time_ms == 50.0
        assert req.max_time_ms == 200.0

    def test_record_request_with_error(self):
        metrics = ServerMetrics()
        metrics.record_request("/v1/audio/transcriptions", 100.0, error=True)

        req = metrics.requests["/v1/audio/transcriptions"]
        assert req.errors == 1

    def test_record_realtime_session_start(self):
        metrics = ServerMetrics()
        metrics.record_realtime_session_start()
        metrics.record_realtime_session_start()

        assert metrics.realtime.active_sessions == 2
        assert metrics.realtime.total_sessions == 2

    def test_record_realtime_session_end(self):
        metrics = ServerMetrics()
        metrics.record_realtime_session_start()
        metrics.record_realtime_session_start()
        metrics.record_realtime_session_end()

        assert metrics.realtime.active_sessions == 1
        assert metrics.realtime.total_sessions == 2

    def test_record_realtime_session_end_never_negative(self):
        metrics = ServerMetrics()
        metrics.record_realtime_session_end()
        assert metrics.realtime.active_sessions == 0

    def test_record_realtime_audio_bytes(self):
        metrics = ServerMetrics()
        metrics.record_realtime_audio_bytes(1000)
        metrics.record_realtime_audio_bytes(500)

        assert metrics.realtime.audio_bytes_received == 1500

    def test_record_realtime_transcription(self):
        metrics = ServerMetrics()
        metrics.record_realtime_transcription()
        metrics.record_realtime_transcription()

        assert metrics.realtime.transcription_events == 2

    def test_get_stats(self):
        metrics = ServerMetrics()
        metrics.record_request("/v1/audio/transcriptions", 100.0)
        metrics.record_request("/v1/audio/transcriptions", 200.0)
        metrics.record_realtime_session_start()
        metrics.record_realtime_audio_bytes(1000)

        stats = metrics.get_stats()

        assert "uptime_seconds" in stats
        assert "requests" in stats
        assert "realtime" in stats
        assert "latency_percentiles" in stats

        assert "/v1/audio/transcriptions" in stats["requests"]
        req_stats = stats["requests"]["/v1/audio/transcriptions"]
        assert req_stats["count"] == 2
        assert req_stats["avg_time_ms"] == 150.0
        assert req_stats["min_time_ms"] == 100.0
        assert req_stats["max_time_ms"] == 200.0

        assert stats["realtime"]["active_sessions"] == 1
        assert stats["realtime"]["audio_bytes_received"] == 1000

    def test_get_stats_latency_percentiles(self):
        metrics = ServerMetrics()
        for i in range(100):
            metrics.record_request("/test", float(i + 1))

        stats = metrics.get_stats()

        assert stats["latency_percentiles"]["p50_ms"] == 51.0
        assert stats["latency_percentiles"]["p95_ms"] == 96.0
        assert stats["latency_percentiles"]["p99_ms"] == 100.0

    def test_get_stats_empty_latency(self):
        metrics = ServerMetrics()
        stats = metrics.get_stats()

        assert stats["latency_percentiles"]["p50_ms"] == 0.0
        assert stats["latency_percentiles"]["p95_ms"] == 0.0
        assert stats["latency_percentiles"]["p99_ms"] == 0.0

    def test_recent_latencies_maxlen(self):
        metrics = ServerMetrics()
        for i in range(150):
            metrics.record_request("/test", float(i))

        assert len(metrics.recent_latencies) == 100

    def test_thread_safety(self):
        import threading

        metrics = ServerMetrics()
        threads = []

        def record_requests():
            for _ in range(100):
                metrics.record_request("/test", 1.0)

        for _ in range(10):
            t = threading.Thread(target=record_requests)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert metrics.requests["/test"].count == 1000
