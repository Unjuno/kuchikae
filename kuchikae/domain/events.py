"""Structured diagnostic events for runtime and benchmark tracing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import time
import uuid
from typing import Any


class EventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class DiagnosticEvent:
    name: str
    level: EventLevel = EventLevel.INFO
    message: str = ""
    run_id: str = ""
    stage: str = ""
    backend: str | None = None
    cache: str | None = None
    elapsed_sec: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
