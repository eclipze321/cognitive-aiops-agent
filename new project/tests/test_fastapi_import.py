"""FastAPI import smoke test.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""

def test_fastapi_app_import():
    from devops_agent.fastapi_app import fastapi_app
    assert fastapi_app.title == 'DevOps Agent FastAPI'
