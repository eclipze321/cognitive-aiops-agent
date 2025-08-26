"""FastAPI wrapper exposing core functionality.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.

Run with:
  uvicorn devops_agent.fastapi_app:fastapi_app --port 8001 --reload
"""
from __future__ import annotations

from fastapi import FastAPI, Depends, Header, HTTPException
from typing import Optional
from .agent import UltimateAIAutonomousDevOps, ALLOWED_KEYS, ADMIN_KEYS
from .schemas import ProcessLogRequest

fastapi_app = FastAPI(title="DevOps Agent FastAPI", version="0.2.0")


def api_key_dep(x_api_key: Optional[str] = Header(default=None)):
    if not x_api_key:
        raise HTTPException(401, detail="missing api key")
    if ALLOWED_KEYS and x_api_key not in ALLOWED_KEYS and x_api_key not in ADMIN_KEYS:
        raise HTTPException(403, detail="invalid api key")
    return x_api_key


@fastapi_app.get('/health')
async def health(_: str = Depends(api_key_dep)):
    from datetime import datetime
    return {'status': 'ok', 'time': datetime.utcnow().isoformat()}


@fastapi_app.post('/process_log')
async def process_log(req: ProcessLogRequest, _: str = Depends(api_key_dep)):
    agent = UltimateAIAutonomousDevOps()
    result = await agent.process_issue(
        req.log, req.mode, req.manual_solution, req.llm_config, req.approvers, req.agents, include_trace=req.include_trace
    )
    agent.close()
    return {'result': result}


@fastapi_app.get('/openapi_custom.json')
async def custom_spec():
    return fastapi_app.openapi()

__all__ = ['fastapi_app']
