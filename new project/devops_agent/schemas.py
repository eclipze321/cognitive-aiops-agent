"""Pydantic request/response schemas for API endpoints.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ProcessLogRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    log: str = Field(min_length=1, description="Raw log text to analyze")
    mode: str = Field(default='safe', description="Execution mode: safe|auto")
    manual_solution: Optional[str] = Field(default=None)
    approvers: Optional[List[str]] = None
    llm_config: Optional[Dict[str, Any]] = None
    agents: Optional[List[str]] = None
    include_trace: bool = True

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in {'safe', 'auto'}:
            raise ValueError('mode must be safe or auto')
        return v


class ThoughtModel(BaseModel):
    step: int
    role: str
    content: str
    score: float | None = None


class ProcessLogResult(BaseModel):
    reasoning: str
    diagnosis_found: bool
    applied: bool
    error_type: Optional[str] = None
    trace: Optional[List[ThoughtModel]] = None
    reasoning_id: str
    request_id: Optional[str] = None


__all__ = [
    'ProcessLogRequest', 'ProcessLogResult', 'ThoughtModel'
]
