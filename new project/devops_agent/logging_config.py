"""Logging configuration utilities.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
import logging
import json
from .config import settings
from contextvars import ContextVar

# Context variables for structured enrichment
request_id_ctx: ContextVar[str | None] = ContextVar('request_id', default=None)
reasoning_id_ctx: ContextVar[str | None] = ContextVar('reasoning_id', default=None)
role_ctx: ContextVar[str | None] = ContextVar('role', default=None)

LOG_FORMAT = '%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s'

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            'ts': self.formatTime(record, '%Y-%m-%dT%H:%M:%S'),
            'level': record.levelname,
            'logger': record.name,
            'msg': record.getMessage(),
        }
        for ctx_var, key in ((request_id_ctx, 'request_id'), (reasoning_id_ctx, 'reasoning_id'), (role_ctx, 'role')):
            val = ctx_var.get()
            if val:
                base[key] = val
        if record.exc_info:
            base['exc'] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

class ContextEnricher(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for ctx_var, key in ((request_id_ctx, 'request_id'), (reasoning_id_ctx, 'reasoning_id'), (role_ctx, 'role')):
            val = ctx_var.get()
            if val:
                setattr(record, key, val)
        return True

class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        parts = [
            f"ts={self.formatTime(record, '%Y-%m-%dT%H:%M:%S')}",
            f"level={record.levelname}",
            f"logger={record.name}",
            f"msg={record.getMessage().replace(' ', '_')}"
        ]
        for attr in ('request_id', 'reasoning_id', 'role'):
            val = getattr(record, attr, None)
            if val:
                parts.append(f"{attr}={val}")
        if record.exc_info:
            parts.append('exc=1')
        return ' '.join(parts)

def configure_logging():
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handlers = []
    handler = logging.StreamHandler()
    if settings.log_format == 'json':
        handler.setFormatter(JsonFormatter())
    elif settings.log_format == 'kv':
        handler.setFormatter(KeyValueFormatter())
    else:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(ContextEnricher())
    handlers.append(handler)
    logging.basicConfig(level=level, handlers=handlers, force=True)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    return logging.getLogger('devops_agent')

def set_log_context(request_id: str | None = None, reasoning_id: str | None = None, role: str | None = None):
    if request_id is not None:
        request_id_ctx.set(request_id)
    if reasoning_id is not None:
        reasoning_id_ctx.set(reasoning_id)
    if role is not None:
        role_ctx.set(role)

__all__ = ['configure_logging', 'set_log_context']
