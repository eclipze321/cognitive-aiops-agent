"""Generate a minimal Python client from the OpenAPI spec.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
from __future__ import annotations

import sys
import requests

SPEC_URL = "http://localhost:5000/api/openapi.json"

TEMPLATE = """# Auto-generated simple client
import requests

class DevOpsAgentClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

    def health(self):
        r = requests.get(f"{self.base_url}/api/health", headers={"X-API-Key": self.api_key})
        r.raise_for_status()
        return r.json()

    def process_log(self, log: str, mode: str = 'safe', include_trace: bool = True):
        payload = {"log": log, "mode": mode, "include_trace": include_trace}
        r = requests.post(f"{self.base_url}/api/process_log", json=payload, headers={"X-API-Key": self.api_key})
        r.raise_for_status()
        return r.json()
"""


def main():
    try:  # best-effort fetch (unused for now, placeholder for future schema-driven generation)
        requests.get(SPEC_URL, timeout=5)
    except Exception:  # pragma: no cover - network issues
        print("Warning: could not fetch spec; generating static client", file=sys.stderr)
    with open('devops_agent_client.py', 'w', encoding='utf-8') as f:
        f.write(TEMPLATE)
    print("Generated devops_agent_client.py")

if __name__ == '__main__':
    main()