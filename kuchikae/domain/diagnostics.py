"""Runtime diagnostic recorder for UI and logs."""

from __future__ import annotations

from collections import deque
import logging
import os
from pathlib import Path

from kuchikae.domain.events import DiagnosticEvent, EventLevel

logger = logging.getLogger(__name__)


class JsonlDiagnosticSink:
    def __init__(self, path: str = "logs/diagnostics.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: DiagnosticEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")


class DiagnosticRecorder:
    def __init__(self, max_events: int = 200, sink: JsonlDiagnosticSink | None = None) -> None:
        self._events: deque[DiagnosticEvent] = deque(maxlen=max_events)
        self._sink = sink or JsonlDiagnosticSink(
            os.environ.get("KUCHIKAE_DIAGNOSTICS_JSONL", "logs/diagnostics.jsonl")
        )

    def emit(self, event: DiagnosticEvent) -> None:
        self._events.append(event)
        logger.info(event.to_json())
        try:
            self._sink.write(event)
        except Exception:
            logger.debug("diagnostic sink write failed", exc_info=True)

    def events(self) -> list[DiagnosticEvent]:
        return list(self._events)

    def user_summary(self) -> str:
        lines: list[str] = []
        for e in self._events:
            if e.level in (EventLevel.WARNING, EventLevel.ERROR) or e.name in {
                "backend.selected",
                "backend.dummy_selected",
                "cache.full_hit",
                "cache.stt_hit",
                "cache.text_hit",
                "cache.voice_hit",
                "pipeline.failed",
            }:
                lines.append(f"[{e.level.value}] {e.stage or e.name}: {e.message}")
        return "\n".join(lines[-20:])
