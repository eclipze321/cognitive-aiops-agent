# DevOps AI Autonomous Agent

Production-ready refactor of original monolithic script.

# DevOps AI Autonomous Agent

Autonomous (but governable) DevOps remediation and reasoning agent. This repository refactors an original monolithic prototype into a production-leaning, modular, testable package with cognitive reasoning scaffold, secure API surfaces, observability, and extensibility.

> Status: Active development (alpha). Interfaces and schemas may evolve; pin a commit for stability.

## Features
## Table of Contents
1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Services](#running-the-services)
7. [API Usage](#api-usage)
8. [Reasoning & Tracing](#reasoning--tracing)
9. [Metrics & Observability](#metrics--observability)
10. [Tracing](#tracing)
11. [Client Generation](#client-generation)
12. [Testing & Quality](#testing--quality)
13. [Security Considerations](#security-considerations)
14. [Docker & Deployment](#docker--deployment)
15. [Roadmap](#roadmap)
16. [Troubleshooting](#troubleshooting)
17. [Contributing](#contributing)
18. [Partnerships / Collaboration](#partnerships--collaboration)
19. [License](#license)


## Quick Start
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
## Installation

### From Source (recommended while alpha)
Clone the repo, then install dependencies:
```powershell
pip install -r requirements.txt
```

### Editable Install
```powershell
pip install -e .
```

### (Future) PyPI
Pending stabilization; planned name: `devops-autonomous-agent`.

## Running the Services

FastAPI (experimental parallel API):
## Configuration
Environment variables (all optional; defaults in `devops_agent/config.py`). Use a `.env` file or set in your runtime environment.

| Variable | Description | Default |
|----------|-------------|---------|
| DB_PATH | SQLite path for knowledge & patterns | devops_knowledge.db |
| LLM_ENDPOINT | External LLM analysis endpoint | http://localhost:8000/llm |
| OPA_ENDPOINT | OPA policy service URL | http://localhost:8080 |
| VAULT_URL | Vault server URL (KV v2 expected) | http://localhost:8200 |
| WEBSOCKET_URL | WebSocket placeholder endpoint | ws://localhost:8765 |
| LOG_LEVEL | Logging level | INFO |
| LOG_FORMAT | text/json/kv | text |
| API_KEYS | Comma list of allowed API keys | None (open) |
| ADMIN_KEYS | Comma list of admin keys | None |
| ENABLE_REFLECTION | Enable reasoning loop | true |
| MAX_REASONING_CYCLES | Cognitive loop max passes | 2 |
| TARGET_REASONING_SCORE | Score threshold to stop | 0.7 |
| RATE_LIMIT_REQUESTS | Requests per window per API key | 100 |
| RATE_LIMIT_WINDOW_SEC | Window size seconds | 60 |
| ENABLE_TRACING | Enable OpenTelemetry spans | false |
| REQUIRE_LICENSE | Enforce license key presence | false |
| LICENSE_SIGNING_SECRET | JWT / hash signing secret | None |

Minimal `.env` example:
```
API_KEYS=sample_key
ADMIN_KEYS=admin_key
ENABLE_REFLECTION=true
ENABLE_TRACING=false
```
## API Usage
Flask app runs on port 5000 (namespace `/api`).

Endpoints:
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health status |
| GET | /api/config | Public config |
| GET | /api/docs | Endpoint list |
| GET | /api/metrics | Prometheus metrics |
| POST | /api/vote_plugin | Vote plugin {plugin, direction} |
| POST | /api/process_log | Process log (preferred) |
| POST | /process_log | Legacy log processing (returns reasoning) |
| GET | /api/openapi.json | Minimal OpenAPI spec |
| GET | /openapi_custom.json (FastAPI) | Full automatic OpenAPI (FastAPI app) |

### Auth & Reasoning Trace
Protected endpoints require an API key via `X-API-Key` header or `?api_key=` parameter. Admin keys (listed in `ADMIN_KEYS`) will be granted elevated role for future admin operations.

Legacy `/process_log` is now also API-key protected if `API_KEYS` is defined.
Use `/api/process_log` going forward; legacy route will be removed in a future release.

Rate limiting: configurable via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_SEC` (in-memory token bucket per API key).

Reflection (heuristic cognitive loop) is enabled by default (`ENABLE_REFLECTION=true`).
Set `enable_reflection` false in env or pass `{ "llm_config": {"strategy": "classic"} }` to disable.
Include a reasoning trace by default; you can disable with `{"include_trace": false}` in `/api/process_log` body.

### Example Request
PowerShell / curl:
```powershell
curl -X POST http://localhost:5000/api/process_log -H "Content-Type: application/json" -H "X-API-Key: your_key" -d '{"log":"Out of memory in module X","mode":"safe"}'
```

Python client snippet:
```python
import requests

payload = {"log": "Out of memory in module X", "mode": "safe", "include_trace": True}
r = requests.post("http://localhost:5000/api/process_log", json=payload, headers={"X-API-Key": "your_key"})
print(r.json())
```

OpenAPI spec: `GET /api/openapi.json` (hand-written minimal). FastAPI auto spec: `GET http://localhost:8001/openapi.json` (when uvicorn running) and interactive docs at `/docs`.

### Metrics
If `prometheus_client` is available, `/api/metrics` serves standard Prometheus exposition; otherwise returns basic counters.

### Licensing (Placeholder)
Enable license requirement by setting `REQUIRE_LICENSE=true` and providing a signed key (future implementation). `LICENSE_SIGNING_SECRET` reserved for signature validation.
If PyJWT is installed (default), license keys can be JWT tokens signed with the secret; fallback hash pattern otherwise.

## Next Steps
...

### Coverage
```powershell
pytest --cov=devops_agent --cov-report=term-missing
```

### Benchmark
```powershell
python -m scripts.benchmark --n 50 --mode safe
```

### Tracing
Set `ENABLE_TRACING=true` to emit OpenTelemetry spans to console exporter (dev use). Production: replace exporter in `tracing.py`.

### Client Generation
Generate a minimal Python client:
```powershell
python -m scripts.gen_client
```
Result: `devops_agent_client.py` with `DevOpsAgentClient` class.

## Partnerships / Collaboration
If you'd like me to partner with your company or project—whether for strategic advisory, joint feature development, pilot deployment, or co-building advanced autonomous DevOps capabilities—please reach out.

How to initiate contact:
1. Open a GitHub Issue titled `Partnership Proposal: <Your Org / Project>` and add the label `partnership`.
2. Include: brief problem statement, current stack (cloud, orchestration, CI/CD, observability), desired outcomes, timeline, and any constraints (compliance, security, SLAs).
3. Optionally provide a secure contact method (email or calendaring link). If confidentiality is required, just note that and we can transition to a private channel.

Sample collaboration themes:
- Embedding autonomous remediation into existing SRE/Platform workflows
- Hardening policy + governance (OPA, audit, RBAC)
- Scaling reasoning loop performance & latency tuning
- Observability + tracing maturity integrations (OTel pipelines, dashboards)
- Plugin / actuator ecosystem extensions (Kubernetes, Terraform, cloud provider ops, security scanners)
- Enterprise readiness: multi-tenancy, tenancy isolation, secrets handling

Early adopter / design partner programs: I'm open to limited deep-engagement slots; propose your use case if you want prioritized influence on roadmap.

Not sure yet? You can still open an issue labeled `discussion` describing challenges—happy to triage.

## License
MIT License. See `LICENSE`.

SPDX-License-Identifier: MIT
```

FastAPI (experimental parallel API):
```powershell
uvicorn devops_agent.fastapi_app:fastapi_app --port 8001 --reload
```

Run with Docker:
```powershell
docker build -t devops-agent .
docker run -p 5000:5000 devops-agent
```

Compose stack (includes OPA & Vault dev):
```powershell
docker compose up --build
```

## Environment Variables
See `devops_agent/config.py` for defaults. Override using a `.env` file.

```
DB_PATH=devops_knowledge.db
LLM_ENDPOINT=http://localhost:8000/llm
OPA_ENDPOINT=http://localhost:8080
VAULT_URL=http://localhost:8200
WEBSOCKET_URL=ws://localhost:8765
LOG_LEVEL=INFO
API_KEYS=public_key_123,public_key_456
ADMIN_KEYS=admin_key_abc
LICENSE_SIGNING_SECRET=change_me
```

## API
Flask app runs on port 5000 (namespace `/api`).

Endpoints:
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health status |
| GET | /api/config | Public config |
| GET | /api/docs | Endpoint list |
| GET | /api/metrics | Prometheus metrics |
| POST | /api/vote_plugin | Vote plugin {plugin, direction} |
| POST | /api/process_log | Process log (preferred) |
| POST | /process_log | Legacy log processing (returns reasoning) |
| GET | /api/openapi.json | Minimal OpenAPI spec |
| GET | /openapi_custom.json (FastAPI) | Full automatic OpenAPI (FastAPI app) |

### Auth & Reasoning Trace
Protected endpoints require an API key via `X-API-Key` header or `?api_key=` parameter. Admin keys (listed in `ADMIN_KEYS`) will be granted elevated role for future admin operations.

Legacy `/process_log` is now also API-key protected if `API_KEYS` is defined.
Use `/api/process_log` going forward; legacy route will be removed in a future release.

Rate limiting: configurable via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_SEC` (in-memory token bucket per API key).

Reflection (heuristic cognitive loop) is enabled by default (`ENABLE_REFLECTION=true`).
Set `enable_reflection` false in env or pass `{ "llm_config": {"strategy": "classic"} }` to disable.
Include a reasoning trace by default; you can disable with `{"include_trace": false}` in `/api/process_log` body.

### Example Request
PowerShell / curl:
```powershell
curl -X POST http://localhost:5000/api/process_log -H "Content-Type: application/json" -H "X-API-Key: your_key" -d '{"log":"Out of memory in module X","mode":"safe"}'
```

Python client snippet:
```python
import requests

payload = {"log": "Out of memory in module X", "mode": "safe", "include_trace": True}
r = requests.post("http://localhost:5000/api/process_log", json=payload, headers={"X-API-Key": "your_key"})
print(r.json())
```

OpenAPI spec: `GET /api/openapi.json` (hand-written minimal). FastAPI auto spec: `GET http://localhost:8001/openapi.json` (when uvicorn running) and interactive docs at `/docs`.

### Metrics
If `prometheus_client` is available, `/api/metrics` serves standard Prometheus exposition; otherwise returns basic counters.

### Licensing (Placeholder)
Enable license requirement by setting `REQUIRE_LICENSE=true` and providing a signed key (future implementation). `LICENSE_SIGNING_SECRET` reserved for signature validation.
If PyJWT is installed (default), license keys can be JWT tokens signed with the secret; fallback hash pattern otherwise.

## Next Steps
- Add unit tests (pytest) for key methods
- Add coverage: `pytest --cov=devops_agent --cov-report=term-missing`
- Implement proper WebSocket server (e.g., FastAPI + websockets / Socket.IO)
- Integrate dependency injection for easier testing
- Add authentication & RBAC for endpoints
- Pin dependency versions for reproducibility
- Add CI pipeline (lint, test, security scan)
	- CI included (GitHub Actions) with lint, type check, coverage

### Coverage
```powershell
pytest --cov=devops_agent --cov-report=term-missing
```

### Benchmark
```powershell
python -m scripts.benchmark --n 50 --mode safe
```

### Tracing
Set `ENABLE_TRACING=true` to emit OpenTelemetry spans to console exporter (dev use). Production: replace exporter in `tracing.py`.

### Client Generation
Generate a minimal Python client:
```powershell
python -m scripts.gen_client
```
Result: `devops_agent_client.py` with `DevOpsAgentClient` class.

## Partnerships / Collaboration
If you'd like me  to partner with your company or project—whether for strategic advisory, joint feature development, pilot deployment, or co-building advanced autonomous DevOps capabilities—please reach out.

How to initiate contact:
1. Open a GitHub Issue titled `Partnership Proposal: <Your Org / Project>` and add the label `partnership`.
2. Include: brief problem statement, current stack (cloud, orchestration, CI/CD, observability), desired outcomes, timeline, and any constraints (compliance, security, SLAs).
3. Optionally provide a secure contact method (email or calendaring link). If confidentiality is required, just note that and we can transition to a private channel.

Sample collaboration themes:
- Embedding autonomous remediation into existing SRE/Platform workflows
- Hardening policy + governance (OPA, audit, RBAC)
- Scaling reasoning loop performance & latency tuning
- Observability + tracing maturity integrations (OTel pipelines, dashboards)
- Plugin / actuator ecosystem extensions (Kubernetes, Terraform, cloud provider ops, security scanners)
- Enterprise readiness: multi-tenancy, tenancy isolation, secrets handling

Early adopter / design partner programs: I'm open to limited deep-engagement slots; propose your use case if you want prioritized influence on roadmap.

Not sure yet? You can still open an issue labeled `discussion` describing challenges—happy to triage.
