"""Pipeline executor for running steps with caching and diagnostics."""

from __future__ import annotations

from typing import Any, Callable

from kuchikae.domain.diagnostics import DiagnosticRecorder
from kuchikae.domain.events import DiagnosticEvent, EventLevel
from kuchikae.domain.error_hints import hint_for_error
from kuchikae.logging import setup_logger
from kuchikae.pipeline.steps import PipelineStep, StepResult

logger = setup_logger("kuchikae.pipeline.executor")


class PipelineExecutor:
    """Executes pipeline steps with caching, diagnostics, and error handling."""

    def __init__(
        self,
        diagnostics: DiagnosticRecorder | None = None,
        run_id: str = "",
    ):
        self._diagnostics = diagnostics or DiagnosticRecorder()
        self._run_id = run_id

    def _emit(
        self,
        name: str,
        message: str,
        stage: str,
        level: EventLevel = EventLevel.INFO,
        backend: str | None = None,
        cache: str | None = None,
        elapsed_sec: float | None = None,
        data: dict | None = None,
    ) -> None:
        self._diagnostics.emit(
            DiagnosticEvent(
                name=name,
                level=level,
                message=message,
                run_id=self._run_id,
                stage=stage,
                backend=backend,
                cache=cache,
                elapsed_sec=elapsed_sec,
                data=data or {},
            )
        )

    def run_step(
        self,
        step: PipelineStep,
        context: dict[str, Any],
        cache_get: Callable[[Any], Any] | None = None,
        cache_set: Callable[[Any, Any], None] | None = None,
        cache_key: Any = None,
        timeout_sec: float = 0,
    ) -> StepResult:
        """Run a step with optional caching."""
        # Check cache
        if cache_get and cache_key is not None:
            cached = cache_get(cache_key)
            if cached is not None:
                logger.info("step:%s:cache_hit", step.name)
                self._emit(
                    f"cache.{step.name}_hit",
                    f"{step.name} cache hit; backend was not executed.",
                    step.name,
                    cache=step.name,
                )
                return StepResult(
                    name=step.name,
                    output=cached,
                    latency_sec=0.0,
                    cached=True,
                )

        # Emit start event
        self._emit(
            f"{step.name}.start",
            f"{step.name} started.",
            step.name,
        )

        # Execute step
        result = step.run(context, timeout_sec=timeout_sec)

        if result.ok:
            # Cache result
            if cache_set and cache_key is not None:
                cache_set(cache_key, result.output)

            # Emit success event
            self._emit(
                f"{step.name}.done",
                f"{step.name} finished.",
                step.name,
                elapsed_sec=result.latency_sec,
            )
        else:
            # Emit failure event
            error_msg = f"{type(result.error).__name__}: {result.error}" if result.error else "Unknown error"
            hint = hint_for_error(step.name, result.error) if result.error else {}
            self._emit(
                f"{step.name}.failed",
                error_msg,
                step.name,
                level=EventLevel.ERROR,
                data={"hint": hint},
            )

        return result

    def run_step_simple(
        self,
        step: PipelineStep,
        context: dict[str, Any],
        timeout_sec: float = 0,
    ) -> StepResult:
        """Run a step without caching (simplified version)."""
        self._emit(
            f"{step.name}.start",
            f"{step.name} started.",
            step.name,
        )

        result = step.run(context, timeout_sec=timeout_sec)

        if result.ok:
            self._emit(
                f"{step.name}.done",
                f"{step.name} finished.",
                step.name,
                elapsed_sec=result.latency_sec,
            )
        else:
            error_msg = f"{type(result.error).__name__}: {result.error}" if result.error else "Unknown error"
            hint = hint_for_error(step.name, result.error) if result.error else {}
            self._emit(
                f"{step.name}.failed",
                error_msg,
                step.name,
                level=EventLevel.ERROR,
                data={"hint": hint},
            )

        return result
