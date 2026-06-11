"""Tests for metrics infrastructure — StreamingLatencyReport + LatencyLogger."""

from __future__ import annotations

import json
import os
import time

import pytest

from kuchikae.domain.metrics import LatencyLogger
from kuchikae.domain.types import (
    PipelineResult,
    StreamChunk,
    StreamingLatencyReport,
)


class TestStreamingLatencyReport:
    """StreamingLatencyReport dataclass."""

    def test_default_construction(self) -> None:
        report = StreamingLatencyReport(session_id="sess_001")
        assert report.session_id == "sess_001"
        assert report.time_to_first_partial_transcript == 0.0
        assert report.time_to_first_committed_transcript == 0.0
        assert report.time_to_first_transformed_text == 0.0
        assert report.time_to_first_audio == 0.0
        assert report.total_recording_sec == 0.0
        assert report.total_processing_sec == 0.0
        assert report.realtime_factor == 0.0
        assert report.timestamp == 0.0
        assert report.stages is None

    def test_realtime_factor_computed(self) -> None:
        report = StreamingLatencyReport(
            session_id="sess_002",
            total_recording_sec=10.0,
            total_processing_sec=5.0,
        )
        assert report.realtime_factor == 0.5

    def test_realtime_factor_zero_recording(self) -> None:
        report = StreamingLatencyReport(
            session_id="sess_003",
            total_recording_sec=0.0,
            total_processing_sec=5.0,
        )
        assert report.realtime_factor == 0.0

    def test_with_stages(self) -> None:
        report = StreamingLatencyReport(
            session_id="sess_004",
            time_to_first_partial_transcript=0.8,
            time_to_first_committed_transcript=1.5,
            time_to_first_transformed_text=2.0,
            time_to_first_audio=2.8,
            total_recording_sec=8.0,
            total_processing_sec=3.0,
            stages={"vad": 0.2, "stt": 0.5, "llm": 0.3, "tts": 0.8},
        )
        assert report.time_to_first_audio == 2.8
        assert report.realtime_factor == 0.375
        assert report.stages is not None
        assert report.stages["vad"] == 0.2


class TestStreamChunk:
    """StreamChunk dataclass."""

    def test_default_construction(self) -> None:
        chunk = StreamChunk(
            session_id="sess_001",
            chunk_index=0,
            audio_start_sec=0.0,
            audio_end_sec=2.0,
        )
        assert chunk.session_id == "sess_001"
        assert chunk.chunk_index == 0
        assert chunk.partial_transcript == ""
        assert chunk.committed_transcript == ""
        assert chunk.output_audio_path is None
        assert chunk.is_final is False

    def test_final_chunk(self) -> None:
        chunk = StreamChunk(
            session_id="sess_002",
            chunk_index=5,
            audio_start_sec=10.0,
            audio_end_sec=12.0,
            partial_transcript="hello world",
            committed_transcript="hello world",
            output_audio_path="/tmp/out.wav",
            is_final=True,
        )
        assert chunk.is_final is True
        assert chunk.output_audio_path == "/tmp/out.wav"
        assert chunk.committed_transcript == "hello world"


class TestLatencyLogger:
    """LatencyLogger JSONL persistence."""

    def test_log_and_read(self, tmp_path) -> None:
        logger = LatencyLogger(log_dir=str(tmp_path))
        report = StreamingLatencyReport(
            session_id="sess_log_001",
            time_to_first_partial_transcript=0.5,
            time_to_first_committed_transcript=1.2,
            time_to_first_transformed_text=1.8,
            time_to_first_audio=2.5,
            total_recording_sec=6.0,
            total_processing_sec=2.0,
        )
        logger.log_report(report)
        assert logger.report_count == 1

        reports = logger.read_reports()
        assert len(reports) == 1
        assert reports[0].session_id == "sess_log_001"
        assert reports[0].realtime_factor == pytest.approx(2.0 / 6.0)

    def test_multiple_reports(self, tmp_path) -> None:
        logger = LatencyLogger(log_dir=str(tmp_path))
        for i in range(3):
            report = StreamingLatencyReport(
                session_id=f"sess_{i:03d}",
                total_recording_sec=5.0,
                total_processing_sec=1.0 + i,
            )
            logger.log_report(report)
        assert logger.report_count == 3

        reports = logger.read_reports()
        assert len(reports) == 3
        assert [r.session_id for r in reports] == ["sess_000", "sess_001", "sess_002"]

    def test_read_empty(self, tmp_path) -> None:
        logger = LatencyLogger(log_dir=str(tmp_path))
        assert logger.read_reports() == []
        assert logger.report_count == 0

    def test_clear(self, tmp_path) -> None:
        logger = LatencyLogger(log_dir=str(tmp_path))
        report = StreamingLatencyReport(session_id="sess_clear")
        logger.log_report(report)
        assert logger.report_count == 1
        logger.clear()
        assert logger.report_count == 0

    def test_timestamp_auto_set(self, tmp_path) -> None:
        logger = LatencyLogger(log_dir=str(tmp_path))
        before = time.time()
        report = StreamingLatencyReport(session_id="sess_ts")
        logger.log_report(report)
        after = time.time()
        reports = logger.read_reports()
        assert before <= reports[0].timestamp <= after

    def test_jsonl_format(self, tmp_path) -> None:
        logger = LatencyLogger(log_dir=str(tmp_path))
        report = StreamingLatencyReport(
            session_id="sess_json",
            time_to_first_audio=1.5,
            total_recording_sec=10.0,
            total_processing_sec=3.0,
        )
        logger.log_report(report)

        log_path = os.path.join(str(tmp_path), "latency.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            line = json.loads(f.readline())
        assert line["session_id"] == "sess_json"
        assert line["time_to_first_audio"] == 1.5
        assert line["realtime_factor"] == 0.3


class TestPipelineResultLatency:
    """PipelineResult now includes latency fields."""

    def test_latency_defaults(self) -> None:
        result = PipelineResult(output_audio_path="/tmp/out.wav")
        assert result.stt_latency == 0.0
        assert result.text_transform_latency == 0.0
        assert result.voice_output_latency == 0.0
        assert result.total_latency == 0.0

    def test_latency_values(self) -> None:
        result = PipelineResult(
            output_audio_path="/tmp/out.wav",
            source_text="hello",
            transformed_text="world",
            stt_latency=0.5,
            text_transform_latency=0.3,
            voice_output_latency=0.8,
            total_latency=1.6,
        )
        assert result.stt_latency == 0.5
        assert result.total_latency == 1.6
