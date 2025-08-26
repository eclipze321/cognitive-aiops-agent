"""Configuration settings.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
from pydantic import Field
try:
    from pydantic_settings import BaseSettings  # Pydantic v2 compatible
except Exception:  # pragma: no cover
    from pydantic import BaseSettings  # type: ignore

class Settings(BaseSettings):
    db_path: str = Field(default='devops_knowledge.db')
    llm_endpoint: str = Field(default='http://localhost:8000/llm')
    opa_endpoint: str = Field(default='http://localhost:8080')
    vault_url: str = Field(default='http://localhost:8200')
    websocket_url: str = Field(default='ws://localhost:8765')
    log_level: str = Field(default='INFO')
    log_format: str = Field(default='text')  # 'text' or 'json'
    license_key: str | None = Field(default=None)
    plugins_dir: str = Field(default='plugins')
    require_license: bool = Field(default=False)
    api_keys: str | None = Field(default=None, description='Comma-separated list of allowed API keys')
    default_role: str = Field(default='reader')
    admin_keys: str | None = Field(default=None, description='Comma-separated list of admin API keys')
    license_signing_secret: str | None = Field(default=None)
    metrics_backend: str = Field(default='prometheus')  # future: 'memory'
    rate_limit_requests: int = Field(default=100)
    rate_limit_window_sec: int = Field(default=60)
    license_jwt_alg: str = Field(default='HS256')
    enable_reflection: bool = Field(default=True)
    max_reasoning_cycles: int = Field(default=2)
    target_reasoning_score: float = Field(default=0.7)
    enable_tracing: bool = Field(default=False)

    class Config:
        env_file = '.env'
        case_sensitive = False

settings = Settings()
