"""Latency metrics logging for Kuchikae streaming pipeline."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

from kuchikae.domain.types import StreamingLatencyReport


class LatencyLogger:
    """JSONL-based latency metrics logger.

    Writes one JSON object per line to ``latency.jsonl`` inside *log_dir*.
    Thread-safe for single-process appends; not designed for concurrent writers.
    """

    def __init__(self, log_dir: str = "latency_logs") -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    @property
    def _path(self) -> str:
        return os.path.join(self.log_dir, "latency.jsonl")

    def log_report(self, report: StreamingLatencyReport) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(report), ensure_ascii=False) + "\n")

    def read_reports(self) -> list[StreamingLatencyReport]:
        if not os.path.exists(self._path):
            return []
        reports: list[StreamingLatencyReport] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                reports.append(StreamingLatencyReport(**data))
        return reports

    def clear(self) -> None:
        if os.path.exists(self._path):
            os.remove(self._path)

    @property
    def report_count(self) -> int:
        return len(self.read_reports())


class StreamingMetricsRecorder:
    """Per-session event-based metrics recorder.

    Records wall-clock timestamps for key streaming events and
    produces a ``StreamingLatencyReport`` on demand.
    Uses ``time.perf_counter`` for high-resolution timing.

    First event wins: calling a ``mark_*`` method more than once
    does **not** overwrite the earlier timestamp.  Pass an explicit
    ``timestamp`` argument for deterministic testing.
    """

    def __init__(self, session_id: str = "") -> None:
        self._session_id = session_id
        self._recording_started_at: float | None = None
        self._first_partial_transcript_at: float | None = None
        self._first_committed_transcript_at: float | None = None
        self._first_transformed_text_at: float | None = None
        self._first_audio_at: float | None = None
        self._recording_finished_at: float | None = None
        self._processing_finished_at: float | None = None

    def mark_recording_started(self, timestamp: float | None = None) -> None:
        if self._recording_started_at is None:
            self._recording_started_at = timestamp if timestamp is not None else time.perf_counter()

    def mark_first_partial_transcript(self, timestamp: float | None = None) -> None:
        if self._first_partial_transcript_at is None:
            self._first_partial_transcript_at = timestamp if timestamp is not None else time.perf_counter()

    def mark_first_committed_transcript(self, timestamp: float | None = None) -> None:
        if self._first_committed_transcript_at is None:
            self._first_committed_transcript_at = timestamp if timestamp is not None else time.perf_counter()

    def mark_first_transformed_text(self, timestamp: float | None = None) -> None:
        if self._first_transformed_text_at is None:
            self._first_transformed_text_at = timestamp if timestamp is not None else time.perf_counter()

    def mark_first_audio(self, timestamp: float | None = None) -> None:
        if self._first_audio_at is None:
            self._first_audio_at = timestamp if timestamp is not None else time.perf_counter()

    def mark_recording_finished(self, timestamp: float | None = None) -> None:
        if self._recording_finished_at is None:
            self._recording_finished_at = timestamp if timestamp is not None else time.perf_counter()

    def mark_processing_finished(self, timestamp: float | None = None) -> None:
        if self._processing_finished_at is None:
            self._processing_finished_at = timestamp if timestamp is not None else time.perf_counter()

    def report(self) -> StreamingLatencyReport:
        return StreamingLatencyReport(
            session_id=self._session_id,
            recording_started_at=self._recording_started_at,
            first_partial_transcript_at=self._first_partial_transcript_at,
            first_committed_transcript_at=self._first_committed_transcript_at,
            first_transformed_text_at=self._first_transformed_text_at,
            first_audio_at=self._first_audio_at,
            recording_finished_at=self._recording_finished_at,
            processing_finished_at=self._processing_finished_at,
        )
