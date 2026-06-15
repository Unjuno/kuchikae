"""Tests for metrics infrastructure — StreamingLatencyReport + LatencyLogger."""

from __future__ import annotations

import json
import os

import pytest

from kuchikae.domain.metrics import LatencyLogger, StreamingMetricsRecorder
from kuchikae.domain.types import (
    PipelineResult,
    StreamChunk,
    StreamingLatencyReport,
)


class TestStreamingLatencyReport:
    """StreamingLatencyReport dataclass with timestamp-based fields."""

    def test_default_construction(self) -> None:
        report = StreamingLatencyReport(session_id="sess_001")
        assert report.session_id == "sess_001"
        assert report.recording_started_at is None
        assert report.first_partial_transcript_at is None

    def test_computed_properties(self) -> None:
        report = StreamingLatencyReport(
            session_id="sess_002",
            recording_started_at=0.0,
            first_partial_transcript_at=0.8,
            first_committed_transcript_at=1.5,
            first_transformed_text_at=2.0,
            first_audio_at=2.8,
            recording_finished_at=8.0,
            processing_finished_at=11.0,
        )
        assert report.time_to_first_partial_transcript == 0.8
        assert report.time_to_first_committed_transcript == 1.5
        assert report.time_to_first_transformed_text == 2.0
        assert report.time_to_first_audio == 2.8
        assert report.recording_duration == 8.0
        assert report.processing_tail_latency == 3.0
        assert report.realtime_factor == pytest.approx(11.0 / 8.0)

    def test_computed_properties_return_none_when_missing(self) -> None:
        report = StreamingLatencyReport(session_id="sess_003")
        assert report.time_to_first_partial_transcript is None
        assert report.recording_duration is None
        assert report.realtime_factor is None

    def test_realtime_factor_none_when_recording_duration_zero(self) -> None:
        report = StreamingLatencyReport(
            session_id="sess_004",
            recording_started_at=0.0,
            recording_finished_at=0.0,
            processing_finished_at=5.0,
        )
        assert report.recording_duration == 0.0
        assert report.realtime_factor is None

    def test_partial_markers(self) -> None:
        """Only some timestamps set; other properties return None."""
        report = StreamingLatencyReport(
            session_id="sess_005",
            recording_started_at=10.0,
            first_partial_transcript_at=10.5,
        )
        assert report.time_to_first_partial_transcript == 0.5
        assert report.time_to_first_audio is None


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
            output_audio_path="out.wav",
            is_final=True,
        )
        assert chunk.is_final is True
        assert chunk.output_audio_path == "out.wav"
        assert chunk.committed_transcript == "hello world"


class TestLatencyLogger:
    """LatencyLogger JSONL persistence."""

    def test_log_and_read(self, tmp_path) -> None:
        logger = LatencyLogger(log_dir=str(tmp_path))
        report = StreamingLatencyReport(
            session_id="sess_log_001",
            recording_started_at=0.0,
            first_partial_transcript_at=0.5,
            recording_finished_at=6.0,
            processing_finished_at=8.0,
        )
        logger.log_report(report)
        assert logger.report_count == 1

        reports = logger.read_reports()
        assert len(reports) == 1
        assert reports[0].session_id == "sess_log_001"

    def test_multiple_reports(self, tmp_path) -> None:
        logger = LatencyLogger(log_dir=str(tmp_path))
        for i in range(3):
            report = StreamingLatencyReport(
                session_id=f"sess_{i:03d}",
                recording_started_at=0.0,
                recording_finished_at=5.0,
                processing_finished_at=6.0 + i,
            )
            logger.log_report(report)
        assert logger.report_count == 3

        reports = logger.read_reports()
        assert len(reports) == 3
        assert [r.session_id for r in reports] == ["sess_000", "sess_001", "sess_002"]

        # Check that realtime_factor round-trips correctly
        assert reports[0].processing_finished_at == 6.0
        assert reports[0].recording_duration == 5.0

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

    def test_jsonl_format(self, tmp_path) -> None:
        logger = LatencyLogger(log_dir=str(tmp_path))
        report = StreamingLatencyReport(
            session_id="sess_json",
            recording_started_at=0.0,
            first_audio_at=1.5,
            recording_finished_at=10.0,
            processing_finished_at=13.0,
        )
        logger.log_report(report)

        log_path = os.path.join(str(tmp_path), "latency.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            line = json.loads(f.readline())
        assert line["session_id"] == "sess_json"
        assert line["first_audio_at"] == 1.5
        assert line["recording_started_at"] == 0.0


class TestStreamingMetricsRecorder:
    """StreamingMetricsRecorder — mark_* methods."""

    def test_mark_and_report(self) -> None:
        rec = StreamingMetricsRecorder(session_id="sess_mr_001")
        rec.mark_recording_started(timestamp=0.0)
        rec.mark_first_partial_transcript(timestamp=0.8)
        rec.mark_first_committed_transcript(timestamp=1.5)
        report = rec.report()
        assert report.session_id == "sess_mr_001"
        assert report.time_to_first_partial_transcript == 0.8
        assert report.time_to_first_committed_transcript == 1.5

    def test_first_event_wins(self) -> None:
        rec = StreamingMetricsRecorder()
        rec.mark_recording_started(timestamp=0.0)
        rec.mark_first_partial_transcript(timestamp=0.8)
        rec.mark_first_partial_transcript(timestamp=0.9)
        report = rec.report()
        assert report.time_to_first_partial_transcript == 0.8

    def test_report_all_none_when_nothing_marked(self) -> None:
        rec = StreamingMetricsRecorder(session_id="sess_mr_002")
        report = rec.report()
        assert report.time_to_first_partial_transcript is None
        assert report.time_to_first_audio is None

    def test_full_lifecycle(self) -> None:
        rec = StreamingMetricsRecorder(session_id="sess_mr_003")
        rec.mark_recording_started(timestamp=0.0)
        rec.mark_first_partial_transcript(timestamp=0.5)
        rec.mark_first_committed_transcript(timestamp=1.0)
        rec.mark_first_transformed_text(timestamp=1.5)
        rec.mark_first_audio(timestamp=2.0)
        rec.mark_recording_finished(timestamp=8.0)
        rec.mark_processing_finished(timestamp=10.0)
        report = rec.report()
        assert report.time_to_first_partial_transcript == 0.5
        assert report.time_to_first_audio == 2.0
        assert report.recording_duration == 8.0
        assert report.realtime_factor == pytest.approx(10.0 / 8.0)

    def test_real_time(self) -> None:
        """Without explicit timestamps, uses perf_counter and produces
        non-negative wall-clock durations."""
        rec = StreamingMetricsRecorder(session_id="sess_mr_real")
        rec.mark_recording_started()
        rec.mark_first_partial_transcript()
        rec.mark_recording_finished()
        report = rec.report()
        assert report.time_to_first_partial_transcript is not None
        assert report.time_to_first_partial_transcript >= 0
        assert report.recording_duration is not None
        assert report.recording_duration >= 0


class TestPipelineResultLatency:
    """PipelineResult now includes latency fields."""

    def test_latency_defaults(self) -> None:
        result = PipelineResult(output_audio_path="out.wav")
        assert result.stt_latency == 0.0
        assert result.text_transform_latency == 0.0
        assert result.voice_output_latency == 0.0
        assert result.total_latency == 0.0

    def test_latency_values(self) -> None:
        result = PipelineResult(
            output_audio_path="out.wav",
            source_text="hello",
            transformed_text="world",
            stt_latency=0.5,
            text_transform_latency=0.3,
            voice_output_latency=0.8,
            total_latency=1.6,
        )
        assert result.stt_latency == 0.5
        assert result.total_latency == 1.6
