"""Schema tests.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
from devops_agent.agent import app
import pytest


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as c:
        yield c


def test_process_log_validation_error(client, monkeypatch):
    monkeypatch.setenv('API_KEYS', 'tkey')
    # invalid mode
    r = client.post('/api/process_log', json={'log': 'abc', 'mode': 'fast'}, headers={'X-API-Key': 'tkey'})
    assert r.status_code == 400
    data = r.get_json()
    assert data['error'] == 'validation_error'

def test_process_log_success_shape(client, monkeypatch):
    monkeypatch.setenv('API_KEYS', 'tkey2')
    r = client.post('/api/process_log', json={'log': 'memory issue detected'}, headers={'X-API-Key': 'tkey2'})
    assert r.status_code == 200
    data = r.get_json()
    assert 'result' in data
    result = data['result']
    assert 'reasoning' in result and 'reasoning_id' in result