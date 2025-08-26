"""Tracing no-op tests.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
from devops_agent.tracing import get_tracer

def test_tracer_noop_when_disabled(monkeypatch):
    monkeypatch.setenv('ENABLE_TRACING', 'false')
    t = get_tracer()
    cm = t.start_as_current_span('x')
    # Should be context manager
    assert hasattr(cm, '__enter__') and hasattr(cm, '__exit__')