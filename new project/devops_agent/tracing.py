"""Optional OpenTelemetry tracing setup.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
from __future__ import annotations

from .config import settings

_TRACER = None

def get_tracer():
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    if not settings.enable_tracing:
        class DummyTracer:  # no-op tracer
            def start_as_current_span(self, name):
                from contextlib import contextmanager
                @contextmanager
                def cm():
                    yield
                return cm()
        _TRACER = DummyTracer()
        return _TRACER
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("devops_agent")
        return _TRACER
    except Exception:  # pragma: no cover
        return get_tracer()  # fallback to dummy

__all__ = ["get_tracer"]