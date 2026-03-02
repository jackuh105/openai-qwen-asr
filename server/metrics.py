import time
from dataclasses import dataclass, field
from typing import Dict, List
from threading import Lock
from collections import deque


@dataclass
class RequestMetrics:
    count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    errors: int = 0

    @property
    def avg_time_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_time_ms / self.count


@dataclass
class RealtimeSessionMetrics:
    active_sessions: int = 0
    total_sessions: int = 0
    audio_bytes_received: int = 0
    transcription_events: int = 0


@dataclass
class ServerMetrics:
    start_time: float = field(default_factory=time.time)
    requests: Dict[str, RequestMetrics] = field(default_factory=lambda: {})
    realtime: RealtimeSessionMetrics = field(default_factory=RealtimeSessionMetrics)
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_request(
        self,
        endpoint: str,
        duration_ms: float,
        error: bool = False,
    ) -> None:
        with self._lock:
            if endpoint not in self.requests:
                self.requests[endpoint] = RequestMetrics()

            metrics = self.requests[endpoint]
            metrics.count += 1
            metrics.total_time_ms += duration_ms

            if duration_ms < metrics.min_time_ms:
                metrics.min_time_ms = duration_ms
            if duration_ms > metrics.max_time_ms:
                metrics.max_time_ms = duration_ms

            if error:
                metrics.errors += 1

            self.recent_latencies.append(duration_ms)

    def record_realtime_session_start(self) -> None:
        with self._lock:
            self.realtime.active_sessions += 1
            self.realtime.total_sessions += 1

    def record_realtime_session_end(self) -> None:
        with self._lock:
            self.realtime.active_sessions = max(0, self.realtime.active_sessions - 1)

    def record_realtime_audio_bytes(self, bytes_count: int) -> None:
        with self._lock:
            self.realtime.audio_bytes_received += bytes_count

    def record_realtime_transcription(self) -> None:
        with self._lock:
            self.realtime.transcription_events += 1

    def get_stats(self) -> dict:
        with self._lock:
            uptime = time.time() - self.start_time

            requests_summary = {}
            for endpoint, metrics in self.requests.items():
                requests_summary[endpoint] = {
                    "count": metrics.count,
                    "avg_time_ms": round(metrics.avg_time_ms, 2),
                    "min_time_ms": round(metrics.min_time_ms, 2)
                    if metrics.min_time_ms != float("inf")
                    else 0,
                    "max_time_ms": round(metrics.max_time_ms, 2),
                    "errors": metrics.errors,
                }

            recent_latencies = list(self.recent_latencies)
            p50 = p95 = p99 = 0.0
            if recent_latencies:
                sorted_latencies = sorted(recent_latencies)
                n = len(sorted_latencies)
                p50 = sorted_latencies[n // 2]
                p95 = sorted_latencies[int(n * 0.95)]
                p99 = sorted_latencies[int(n * 0.99)]

            return {
                "uptime_seconds": round(uptime, 2),
                "requests": requests_summary,
                "realtime": {
                    "active_sessions": self.realtime.active_sessions,
                    "total_sessions": self.realtime.total_sessions,
                    "audio_bytes_received": self.realtime.audio_bytes_received,
                    "transcription_events": self.realtime.transcription_events,
                },
                "latency_percentiles": {
                    "p50_ms": round(p50, 2),
                    "p95_ms": round(p95, 2),
                    "p99_ms": round(p99, 2),
                },
            }


metrics = ServerMetrics()
