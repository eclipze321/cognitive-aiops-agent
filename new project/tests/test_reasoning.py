"""Reasoning tests.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
import pytest
from devops_agent.agent import UltimateAIAutonomousDevOps

@pytest.fixture
def agent():
    a = UltimateAIAutonomousDevOps()
    yield a
    a.close()

def test_explain_reasoning_unknown(agent):
    info = agent.analyze_log('unrecognized pattern message')
    text = agent.explain_reasoning(info, None)
    assert 'No solution found' in text

def test_reasoning_trace(agent):
    data = agent.reason_about_log('mysterious outage code 9999')
    assert 'error_type' in data
    assert hasattr(agent, '_last_thoughts')
    assert isinstance(agent._last_thoughts, list)
