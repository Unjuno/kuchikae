"""Latency metrics logging for Kuchikae streaming pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

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
        if report.timestamp == 0.0:
            report.timestamp = datetime.now(timezone.utc).timestamp()
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(report), ensure_ascii=False) + "\n")

    def read_reports(self) -> list[StreamingLatencyReport]:
        if not os.path.exists(self._path):
            return []
        reports: list[StreamingLatencyReport] = []
        with open(self._path) as f:
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
