# Copyright (c) 2025 Patrick Morrison
# Licensed under the MIT License.
from devops_agent.agent import UltimateAIAutonomousDevOps


def test_log_pattern_detection():
    agent = UltimateAIAutonomousDevOps()
    info = agent.analyze_log('Out of memory error occurred in process 1234')
    assert info['error_type'] == 'memory'
    agent.close()


def test_no_pattern():
    agent = UltimateAIAutonomousDevOps()
    info = agent.analyze_log('Completely unknown message')
    assert info['error_type'] == 'unknown'
    agent.close()

def test_cognitive_reasoning():
    agent = UltimateAIAutonomousDevOps()
    info = agent.reason_about_log('Completely unknown message with code ZXY')
    assert 'error_type' in info
    assert 'summary' in info
    agent.close()
