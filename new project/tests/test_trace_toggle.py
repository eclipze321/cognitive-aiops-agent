"""Trace toggle tests.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
import pytest
from devops_agent.agent import app


@pytest.fixture
def client(monkeypatch):
    app.testing = True
    monkeypatch.setenv('API_KEYS', 'ktrace')
    with app.test_client() as c:
        yield c


def test_trace_default_present(client):
    r = client.post('/api/process_log', json={'log': 'memory error happening'}, headers={'X-API-Key': 'ktrace'})
    data = r.get_json()['result']
    assert 'trace' in data


def test_trace_disabled(client):
    r = client.post('/api/process_log', json={'log': 'memory error happening', 'include_trace': False}, headers={'X-API-Key': 'ktrace'})
    data = r.get_json()['result']
    assert 'trace' not in data
