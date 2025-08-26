"""Rate limit tests.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
import pytest
from devops_agent.agent import app, _RATE_BUCKET, settings


@pytest.fixture
def client(monkeypatch):
    app.testing = True
    # tighten limit
    monkeypatch.setattr(settings, 'rate_limit_requests', 2)
    with app.test_client() as c:
        _RATE_BUCKET.clear()
        yield c


def test_rate_limit_trigger(client, monkeypatch):
    monkeypatch.setenv('API_KEYS', 'krl')
    headers = {'X-API-Key': 'krl'}
    for _ in range(2):
        r = client.get('/api/health', headers=headers)
        assert r.status_code == 200
    r3 = client.get('/api/health', headers=headers)
    assert r3.status_code == 429
