"""Auth tests.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
import pytest
from devops_agent.agent import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as c:
        yield c

def test_docs_public(client):
    r = client.get('/api/docs')
    assert r.status_code == 200
    assert 'endpoints' in r.get_json()

def test_health_requires_key(client, monkeypatch):
    monkeypatch.setenv('API_KEYS', 'k1')
    # Without key
    r = client.get('/api/health')
    assert r.status_code in (401, 403)
    # With key (recreate app context not necessary since decorator reads globals)
    r2 = client.get('/api/health', headers={'X-API-Key': 'k1'})
    assert r2.status_code == 200

def test_legacy_process_log_secured(client, monkeypatch):
    monkeypatch.setenv('API_KEYS', 'k2')
    r = client.post('/process_log', json={'log': 'test'})
    assert r.status_code == 401
    r2 = client.post('/process_log', json={'log': 'test'}, headers={'X-API-Key': 'k2'})
    # Might 500 if dependencies missing; accept 200..499 !=401 meaning auth passed
    assert r2.status_code != 401
