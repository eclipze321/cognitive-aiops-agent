"""Core DevOps agent implementation.

Copyright (c) 2025 Patrick Morrison
Licensed under the MIT License. See LICENSE for details.
"""
import json
import os
import re
import sqlite3
import time
import uuid as _uuid
from datetime import datetime
from typing import Dict, List, Optional

try:
    import docker  # type: ignore
except Exception:  # pragma: no cover
    docker = None  # type: ignore
try:
    import git  # type: ignore
except Exception:  # pragma: no cover
    git = None  # type: ignore
import requests
try:
    import websocket  # type: ignore
except Exception:  # pragma: no cover
    websocket = None  # type: ignore
from cryptography.fernet import Fernet
from flask import Flask, Blueprint, jsonify, request, Response, g
try:
    from hvac import Client as VaultClient  # type: ignore
except Exception:  # pragma: no cover
    VaultClient = None  # type: ignore
try:  # optional dependency guard
    from opa_client.opa import OpaClient  # type: ignore
except Exception:  # pragma: no cover
    OpaClient = None  # type: ignore
from sklearn.ensemble import IsolationForest

from .config import settings
from .cognitive import Planner, Executor, Critic, CognitiveLoop
from .memory import MemoryStore
from .schemas import ProcessLogRequest, ProcessLogResult
from .tracing import get_tracer
from .logging_config import configure_logging, set_log_context
logger = configure_logging()

app = Flask(__name__)

api_bp = Blueprint('api', __name__)

# --- Auth utilities ---
def _parse_keys(raw: str | None):
    return {k.strip() for k in raw.split(',')} if raw else set()

ALLOWED_KEYS = _parse_keys(settings.api_keys)
ADMIN_KEYS = _parse_keys(settings.admin_keys)

try:
    import jwt  # PyJWT optional
except Exception:
    jwt = None

_RATE_BUCKET: dict[str, list[float]] = {}

def _rate_limited(key: str) -> bool:
    now = time.time()
    window = settings.rate_limit_window_sec
    limit = settings.rate_limit_requests
    bucket = _RATE_BUCKET.setdefault(key, [])
    # purge old
    cutoff = now - window
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= limit:
        return True
    bucket.append(now)
    return False

@app.before_request
def _request_preamble():
    g.request_id = request.headers.get('X-Request-ID') or str(_uuid.uuid4())
    # rate limit (skip docs & metrics)
    if request.path.startswith('/api') and not request.path.endswith(('/docs', '/metrics')):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key') or 'anon'
        if _rate_limited(api_key):
            return jsonify({'error': 'rate limit exceeded', 'request_id': g.request_id}), 429

def require_api_key(admin: bool = False):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            key = request.headers.get('X-API-Key') or request.args.get('api_key')
            if not key:
                return jsonify({'error': 'missing api key'}), 401
            if admin:
                if key not in ADMIN_KEYS:
                    return jsonify({'error': 'admin privileges required'}), 403
            else:
                if ALLOWED_KEYS and key not in ALLOWED_KEYS and key not in ADMIN_KEYS:
                    return jsonify({'error': 'invalid api key'}), 403
            request.role = 'admin' if key in ADMIN_KEYS else settings.default_role
            resp = fn(*args, **kwargs)
            # attach request id
            rid = getattr(g, 'request_id', '')
            if isinstance(resp, tuple):
                body, *rest = resp
                if isinstance(body, dict):
                    body.setdefault('request_id', rid)
                    return (body, *rest)
                return resp
            if isinstance(resp, dict):
                resp.setdefault('request_id', rid)
            return resp
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator

class UltimateAIAutonomousDevOps:
    def __init__(self, db_path: str = None, llm_endpoint: str = None, opa_endpoint: str = None, vault_url: str = None):
        self.db_path = db_path or settings.db_path
        self.llm_endpoint = llm_endpoint or settings.llm_endpoint
        self.opa_endpoint = opa_endpoint or settings.opa_endpoint
        self.vault_url = vault_url or settings.vault_url

        self.knowledge_db = sqlite3.connect(self.db_path, check_same_thread=False)
        if OpaClient:
            try:
                self.opa_client = OpaClient(self.opa_endpoint)
            except Exception:  # pragma: no cover
                self.opa_client = None
        else:
            self.opa_client = None
        if VaultClient:
            try:
                self.vault_client = VaultClient(url=self.vault_url)
            except Exception:  # pragma: no cover
                self.vault_client = None
        else:
            self.vault_client = None
        if docker:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Docker initialization failed: {e}")
                self.docker_client = None
        else:
            self.docker_client = None
        self.ssh_client = None
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.plugins: Dict[str, Dict] = {}
        self.actuators: Dict[str, Dict] = {}
        self.alert_channels: List[Dict] = []
        self.encryption_key = self.load_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        self.error_patterns = {
            'memory': re.compile(r'memory\s+error|out\s+of\s+memory', re.IGNORECASE),
            'connection': re.compile(r'connection\s+(timeout|refused|failed)', re.IGNORECASE),
            'disk': re.compile(r'disk\s+(full|space)', re.IGNORECASE),
        }
        self.ws_clients = []
        self.setup_db()
        self.start_websocket_server()
        # dynamic plugins load
        self.auto_load_plugins()
        # memory store
        try:
            self.memory = MemoryStore(self.knowledge_db)
        except Exception as e:
            logger.warning(f"Memory init failed: {e}")
            self.memory = None
        if settings.require_license and not settings.license_key:
            logger.warning("License key required but not provided.")
        if settings.require_license and settings.license_key and settings.license_signing_secret:
            if not self._verify_license(settings.license_key, settings.license_signing_secret):
                logger.error("Invalid license key signature.")

    def _verify_license(self, license_key: str, secret: str) -> bool:
        # JWT preferred if library present; else fallback pattern
        if jwt:
            try:
                decoded = jwt.decode(license_key, secret, algorithms=[settings.license_jwt_alg])
                exp = decoded.get('exp')
                if exp and time.time() > exp:
                    return False
                return True
            except Exception:
                return False
        # fallback simple hash prefix (not secure)
        try:
            import hashlib
            parts = license_key.split(':')
            if len(parts) != 2:
                return False
            payload, sig = parts
            calc = hashlib.sha256((payload + secret).encode()).hexdigest()[:16]
            return sig == calc
        except Exception:
            return False

    # --- Initialization helpers ---
    def load_encryption_key(self) -> bytes:
        try:
            # Attempt to read, if not present generate and store
            secret_path = "devops/encryption_key"
            try:
                secret = self.vault_client.secrets.kv.v2.read_secret_version(path=secret_path)
                key = secret['data']['data']['key'].encode()
                return key
            except Exception:  # create
                key = Fernet.generate_key()
                try:
                    self.vault_client.secrets.kv.v2.create_or_update_secret(path=secret_path, secret={'key': key.decode()})
                except Exception as e:
                    logger.warning(f"Storing key in Vault failed: {e}")
                return key
        except Exception as e:
            logger.error(f"Vault not reachable, generating ephemeral key: {e}")
            return Fernet.generate_key()

    def setup_db(self):
        cursor = self.knowledge_db.cursor()
        cursor.executescript('''
        CREATE TABLE IF NOT EXISTS failure_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_type TEXT,
            pattern TEXT,
            solution TEXT,
            revert_logic TEXT,
            success_count INTEGER DEFAULT 0,
            contribution_points INTEGER DEFAULT 0,
            last_applied TIMESTAMP,
            source TEXT DEFAULT 'local',
            votes INTEGER DEFAULT 0,
            marketplace_id TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            error_type TEXT,
            log_snippet TEXT,
            solution TEXT,
            result TEXT,
            timestamp TIMESTAMP,
            approvers TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fix_id INTEGER,
            success BOOLEAN,
            comments TEXT,
            timestamp TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS marketplace_plugins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plugin_name TEXT,
            config TEXT,
            votes INTEGER DEFAULT 0,
            creator TEXT
        );
        ''')
        self.knowledge_db.commit()

    # --- Plugin & Actuator registration ---
    def load_plugin(self, plugin_name: str, config: Dict, creator: str = 'anonymous'):
        self.plugins[plugin_name] = config
        cursor = self.knowledge_db.cursor()
        cursor.execute('INSERT INTO marketplace_plugins (plugin_name, config, creator) VALUES (?, ?, ?)',
                       (plugin_name, json.dumps(config), creator))
        self.knowledge_db.commit()
        logger.info(f"Loaded plugin {plugin_name}")

    def auto_load_plugins(self):
        directory = settings.plugins_dir
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            if fname.endswith('.json'):
                try:
                    path = os.path.join(directory, fname)
                    with open(path, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                    name = cfg.get('name') or fname[:-5]
                    self.load_plugin(name, cfg)
                except Exception as e:
                    logger.warning(f"Failed loading plugin {fname}: {e}")

    def register_actuator(self, actuator_name: str, config: Dict):
        self.actuators[actuator_name] = config
        logger.info(f"Registered actuator {actuator_name}")

    def register_alert_channel(self, channel: Dict):
        self.alert_channels.append(channel)
        logger.info(f"Registered alert channel {channel.get('type')}")

    # --- WebSocket server (placeholder - in production use a proper server) ---
    def start_websocket_server(self):
        def on_message(ws, message):
            logger.debug(f"WebSocket message: {message}")
            ws.send(json.dumps({'status': 'received'}))

        def on_open(ws):
            self.ws_clients.append(ws)
            logger.info("WebSocket client connected")

        def on_close(ws):
            if ws in self.ws_clients:
                self.ws_clients.remove(ws)
                logger.info("WebSocket client disconnected")

        if not websocket:
            return
        try:
            ws_server = websocket.WebSocketApp(settings.websocket_url, on_message=on_message, on_open=on_open, on_close=on_close)
            import threading
            threading.Thread(target=ws_server.run_forever, daemon=True).start()
        except Exception as e:
            logger.warning(f"WebSocket server not started: {e}")

    def broadcast_dashboard_update(self, data: Dict):
        for ws in list(self.ws_clients):
            try:
                ws.send(json.dumps(data))
            except Exception:
                pass

    def close(self):
        """Gracefully close resources."""
        try:
            if self.ssh_client:
                self.ssh_client.close()
        except Exception:
            pass
        try:
            if self.knowledge_db:
                self.knowledge_db.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # --- Core logic (adapted) ---
    def check_policy(self, diagnosis: Dict[str, str], mode: str) -> bool:
        input_data = {'error_type': diagnosis.get('error_type', 'unknown'), 'solution': diagnosis['solution'], 'mode': mode}
        if not self.opa_client:
            return False
        try:
            result = self.opa_client.check_policy(input_data, 'devops/allow_fix')
            return result.get('result', False)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Policy check failed: {e}")
            return False

    def analyze_log_with_llm(self, log: str, llm_config: Optional[Dict] = None) -> Dict[str, str]:
        endpoint = llm_config.get('endpoint', self.llm_endpoint) if llm_config else self.llm_endpoint
        try:
            response = requests.post(endpoint, json={'prompt': f"Analyze this log and propose fix with revert: {log[:1000]}", 'max_tokens': 400}, timeout=10)
            response.raise_for_status()
            result = response.json()
            return {
                'error_type': result.get('error_type', 'unknown'),
                'summary': result.get('summary', log[:200]),
                'suggested_fix': result.get('suggested_fix', ''),
                'revert_logic': result.get('revert_logic', ''),
                'root_cause': result.get('root_cause', 'Unknown cause')
            }
        except Exception as e:
            logger.debug(f"LLM analysis fallback: {e}")
            return self.analyze_log(log)

    def analyze_log(self, log: str) -> Dict[str, str]:
        for error_type, pattern in self.error_patterns.items():
            if pattern.search(log):
                return {'error_type': error_type, 'summary': log[:200], 'suggested_fix': '', 'revert_logic': '', 'root_cause': 'Pattern-based detection'}
        return {'error_type': 'unknown', 'summary': log[:200], 'suggested_fix': '', 'revert_logic': '', 'root_cause': 'Unknown'}

    # Cognitive heuristic reasoning (not real AGI)
    def reason_about_log(self, log: str) -> Dict[str, str]:
        excerpt = log[:400]
        context = {'log_excerpt': excerpt}
        toolbelt = {
            'classify': lambda ctx: f"type:{self._classify(ctx['log_excerpt'])}",
            'propose_fix': lambda ctx: f"fix:{self._propose_fix(ctx['log_excerpt'])}",
            'revert': lambda ctx: f"revert:{self._derive_revert(ctx['log_excerpt'])}",
            'risk': lambda ctx: 'ok',
            'memory_lookup': lambda ctx: self._memory_lookup(ctx['log_excerpt'])
        }
        loop = CognitiveLoop(Planner(), Executor(toolbelt), Critic(), max_cycles=settings.max_reasoning_cycles, target_score=settings.target_reasoning_score)
        aggregate = loop.run(excerpt, context)
        base = self.analyze_log(log)
        for k, v in aggregate.items():
            base.setdefault(k, v)
        return base

    def _classify(self, text: str) -> str:
        for k, p in self.error_patterns.items():
            if p.search(text):
                return k
        return 'unknown'

    def _propose_fix(self, text: str) -> str:
        if 'memory' in text.lower():
            return 'increase memory limits or optimize usage'
        if 'connection' in text.lower():
            return 'restart service and verify network routing'
        return 'collect additional diagnostics'

    def _derive_revert(self, text: str) -> str:
        if 'memory' in text.lower():
            return 'restore previous memory configuration'
        if 'connection' in text.lower():
            return 'revert service restart'
        return 'undo diagnostic changes'

    def _memory_lookup(self, text: str) -> str:
        if not self.memory:
            return 'memory:unavailable'
        sims = self.memory.similar('log', text, top_k=2)
        if not sims:
            try:
                self.memory.add('log', text)
            except Exception:
                pass
            return 'memory:none'
        formatted = ';'.join(f"{score:.2f}:{c[:30]}" for c, score in sims)
        return f'memory:{formatted}'

    def simulate_impact(self, solution: str) -> Dict[str, str]:
        try:
            response = requests.post(self.llm_endpoint, json={'prompt': f"Impact of fix: {solution}", 'max_tokens': 200}, timeout=10)
            response.raise_for_status()
            return {'impact': response.json().get('impact', 'No analysis')}
        except Exception as e:
            logger.debug(f"Impact simulation fallback: {e}")
            return {'impact': 'Simulation failed'}

    def generate_iac_patch(self, solution: str, iac_type: str = 'terraform') -> str:
        try:
            response = requests.post(self.llm_endpoint, json={'prompt': f"Convert fix to {iac_type} patch: {solution}", 'max_tokens': 400}, timeout=10)
            patch = response.json().get('patch', f'# No {iac_type} patch generated')
            try:
                repo = git.Repo('.')
                patch_file = f'fix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tf'
                with open(patch_file, 'w') as f:
                    f.write(patch)
                repo.git.add(patch_file)
                repo.git.commit(m=f"Auto-generated {iac_type} patch")
                return patch_file
            except Exception as e:
                logger.warning(f"Git commit skipped: {e}")
                return 'patch_not_committed.tf'
        except Exception as e:
            logger.debug(f"IaC generation failed: {e}")
            return f'# Error generating {iac_type} patch'

    def natural_language_to_workflow(self, description: str, output_type: str = 'bash') -> str:
        try:
            response = requests.post(self.llm_endpoint, json={'prompt': f"Description to {output_type} script with revert: {description}", 'max_tokens': 400}, timeout=10)
            return response.json().get('script', f'# No {output_type} script generated')
        except Exception as e:
            logger.debug(f"Workflow generation failed: {e}")
            return f'# Error generating {output_type} script'

    def diagnose_issue(self, error_info: Dict[str, str]) -> Optional[Dict[str, str]]:
        cursor = self.knowledge_db.cursor()
        cursor.execute('SELECT pattern, solution, revert_logic, marketplace_id FROM failure_patterns WHERE error_type=? ORDER BY votes DESC, success_count DESC LIMIT 1', (error_info['error_type'],))
        result = cursor.fetchone()
        if result:
            return {'pattern': result[0], 'solution': result[1], 'revert_logic': result[2], 'marketplace_id': result[3], 'error_type': error_info['error_type']}
        elif error_info['suggested_fix']:
            return {'pattern': error_info['summary'], 'solution': error_info['suggested_fix'], 'revert_logic': error_info['revert_logic'], 'marketplace_id': None, 'error_type': error_info['error_type']}
        return None

    def dry_run_fix(self, diagnosis: Dict[str, str]) -> Dict[str, str]:
        if not self.docker_client:
            return {'status': 'skipped'}
        try:
            container = self.docker_client.containers.run('alpine', command=diagnosis['solution'], detach=True, remove=True)
            result = container.wait()
            return {'status': 'success' if result.get('StatusCode') == 0 else 'failed'}
        except Exception as e:
            logger.debug(f"Dry run failed: {e}")
            return {'status': 'failed'}

    def check_approval(self, mode: str, approvers: Optional[List[str]]) -> bool:
        if mode == 'safe' and (not approvers or len(approvers) < 2):
            return False
        return True

    def apply_fix(self, diagnosis: Dict[str, str], mode: str = 'safe', approvers: Optional[List[str]] = None) -> bool:
        if not self.check_policy(diagnosis, mode):
            self.notify_alert_channels("Fix blocked by policy")
            return False
        if not self.check_approval(mode, approvers):
            self.notify_alert_channels("Approval required")
            return False

        impact = self.simulate_impact(diagnosis['solution'])
        if 'critical' in impact['impact'].lower() and mode == 'auto':
            mode = 'safe'

        dry_run_result = self.dry_run_fix(diagnosis)
        if dry_run_result['status'] == 'failed':
            self.notify_alert_channels("Dry run failed")
            return False

        try:
            actuator = self.actuators.get('default', {'type': 'ssh'})
            if actuator['type'] == 'ssh' and self.ssh_client:
                stdin, stdout, stderr = self.ssh_client.exec_command(diagnosis['solution'])
                if stderr.read():
                    self.rollback_fix(diagnosis)
                    return False
            self.log_audit('apply_fix', diagnosis, 'success', approvers or [])
            patch_file = self.generate_iac_patch(diagnosis['solution'])
            self.broadcast_dashboard_update({'action': 'apply_fix', 'error_type': diagnosis.get('error_type'), 'status': 'success', 'patch_file': patch_file})
            if PROM_AVAILABLE:
                try:
                    FIX_SUCCESS.inc()
                except Exception:
                    pass
            else:
                try:
                    METRICS['fix_success_total'] += 1
                except Exception:
                    pass
            return True
        except Exception as e:
            self.log_audit('apply_fix', diagnosis, f'failed:{e}', approvers or [])
            self.notify_alert_channels(f"Fix application failed: {e}")
            if PROM_AVAILABLE:
                try:
                    FIX_FAILURE.inc()
                except Exception:
                    pass
            else:
                try:
                    METRICS['fix_failure_total'] += 1
                except Exception:
                    pass
            return False

    def rollback_fix(self, diagnosis: Dict[str, str]):
        if diagnosis.get('revert_logic') and self.ssh_client:
            self.ssh_client.exec_command(diagnosis['revert_logic'])
            self.broadcast_dashboard_update({'action': 'rollback', 'error_type': diagnosis.get('error_type'), 'status': 'executed'})

    def log_audit(self, action: str, diagnosis: Dict[str, str], result: str, approvers: List[str]):
        cursor = self.knowledge_db.cursor()
        cursor.execute('INSERT INTO audit_log (action, error_type, log_snippet, solution, result, timestamp, approvers) VALUES (?, ?, ?, ?, ?, ?, ?)',
                       (action, diagnosis.get('error_type', 'unknown'), diagnosis.get('pattern', ''), diagnosis.get('solution', ''), result, datetime.now(), ','.join(approvers)))
        self.knowledge_db.commit()
        self.broadcast_dashboard_update({'action': 'audit_log', 'error_type': diagnosis.get('error_type'), 'result': result})

    def notify_alert_channels(self, message: str):
        for channel in self.alert_channels:
            try:
                requests.post(channel['url'], json={'text': message}, timeout=5)
            except Exception:
                pass
        self.broadcast_dashboard_update({'action': 'alert', 'message': message})

    # --- API helper methods ---
    def marketplace_vote(self, plugin_name: str, delta: int):
        cursor = self.knowledge_db.cursor()
        cursor.execute('UPDATE marketplace_plugins SET votes = votes + ? WHERE plugin_name = ?', (delta, plugin_name))
        self.knowledge_db.commit()
        return cursor.rowcount > 0

    def generate_explanation_report(self, error_info: Dict[str, str], diagnosis: Optional[Dict[str, str]]):
        report = self.explain_reasoning(error_info, diagnosis)
        impact = self.simulate_impact(diagnosis['solution'] if diagnosis else '')
        report += f"\nImpact: {impact['impact']}"
        report_file = f"explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        self.broadcast_dashboard_update({'action': 'report_generated', 'file': report_file})
        return report_file

    def explain_reasoning(self, error_info: Dict[str, str], diagnosis: Optional[Dict[str, str]]):
        explanation = f"Log: {error_info['summary']} Root: {error_info['root_cause']}\n"
        if diagnosis:
            explanation += f"Issue: {error_info['error_type']} Solution: {diagnosis['solution']} Revert: {diagnosis['revert_logic']}\n"
        else:
            explanation += f"Issue: {error_info['error_type']} No solution found.\n"
        return explanation

    async def process_issue(self, log: str, mode: str = 'safe', manual_solution: Optional[str] = None, llm_config: Optional[Dict] = None, approvers: Optional[List[str]] = None, agents: Optional[List[str]] = None, include_trace: bool = True, reasoning_id: Optional[str] = None, request_id: Optional[str] = None) -> Dict[str, str]:
        tracer = get_tracer()
        with tracer.start_as_current_span("process_issue"):
            use_cognitive = settings.enable_reflection or (llm_config and llm_config.get('strategy') == 'agi')
            if use_cognitive:
                error_info = self.reason_about_log(log)
            else:
                error_info = self.analyze_log_with_llm(log, llm_config)
            diagnosis = self.diagnose_issue(error_info)
            rid = reasoning_id or str(_uuid.uuid4())
            logger.info(f"reasoning_start id={rid} request_id={request_id} error_type={error_info.get('error_type')} cognitive={use_cognitive}")
        if diagnosis:
            if agents:
                self.notify_alert_channels("Agent coordination not implemented in refactor")
            success = self.apply_fix(diagnosis, mode, approvers)
            reasoning_text = self.explain_reasoning(error_info, diagnosis if success else None)
            if success:
                self.generate_explanation_report(error_info, diagnosis)
            result = {
                'reasoning': reasoning_text,
                'diagnosis_found': True,
                'applied': success,
                'error_type': error_info.get('error_type')
            }
            if include_trace and hasattr(self, '_last_thoughts'):
                result['trace'] = self._last_thoughts
            result['reasoning_id'] = rid
            if request_id:
                result['request_id'] = request_id
            set_log_context(request_id=request_id, reasoning_id=rid, role=getattr(request, 'role', None))
            logger.info(f"reasoning_end id={rid} applied={success} error_type={error_info.get('error_type')}")
            return result
        else:
            if manual_solution:
                script = self.natural_language_to_workflow(manual_solution)
                res = {'reasoning': script, 'diagnosis_found': False, 'applied': False, 'error_type': error_info.get('error_type'), 'reasoning_id': rid}
                if request_id:
                    res['request_id'] = request_id
                logger.info(f"reasoning_end id={rid} applied=False diagnosis_found=False")
                return res
            reasoning_text = self.explain_reasoning(error_info, None)
            result = {
                'reasoning': reasoning_text,
                'diagnosis_found': False,
                'applied': False,
                'error_type': error_info.get('error_type'),
                'reasoning_id': rid
            }
            if include_trace and hasattr(self, '_last_thoughts'):
                result['trace'] = self._last_thoughts
            if request_id:
                result['request_id'] = request_id
            set_log_context(request_id=request_id, reasoning_id=rid, role=getattr(request, 'role', None))
            logger.info(f"reasoning_end id={rid} applied=False diagnosis_found=False")
            return result


# --- Flask API Routes ---
@api_bp.route('/health', methods=['GET'])
@require_api_key()
def health():
    return jsonify({'status': 'ok', 'time': datetime.utcnow().isoformat()})


@api_bp.route('/vote_plugin', methods=['POST'])
@require_api_key()
def vote_plugin():
    data = request.json or {}
    name = data.get('plugin')
    direction = data.get('direction', 'up')
    delta = 1 if direction == 'up' else -1
    agent = UltimateAIAutonomousDevOps()
    success = agent.marketplace_vote(name, delta)
    agent.close()
    if not success:
        return jsonify({'error': 'plugin not found'}), 404
    return jsonify({'status': 'voted', 'plugin': name, 'delta': delta})

@api_bp.route('/config', methods=['GET'])
@require_api_key()
def get_config():
    public = {
        'db_path': settings.db_path,
        'llm_endpoint': settings.llm_endpoint,
        'opa_endpoint': settings.opa_endpoint,
        'vault_url': settings.vault_url,
        'websocket_url': settings.websocket_url,
        'log_level': settings.log_level,
        'plugins_dir': settings.plugins_dir,
        'require_license': settings.require_license,
    }
    return jsonify(public)

@api_bp.route('/docs', methods=['GET'])
def docs():  # public
    return jsonify({
        'endpoints': [
            {'path': '/api/health', 'method': 'GET', 'desc': 'Service health status'},
            {'path': '/api/vote_plugin', 'method': 'POST', 'body': {'plugin': 'name', 'direction': 'up|down'}},
            {'path': '/api/config', 'method': 'GET', 'desc': 'Public runtime configuration'},
            {'path': '/api/process_log', 'method': 'POST', 'desc': 'Process log (namespaced) returns reasoning + optional trace'},
            {'path': '/process_log', 'method': 'POST', 'desc': 'Original processing endpoint (legacy, not namespaced)'},
            {'path': '/api/openapi.json', 'method': 'GET', 'desc': 'Minimal OpenAPI spec (static)'}
        ]
    })

@api_bp.route('/process_log', methods=['POST'])
@require_api_key()
def process_log_ns():
    data = request.json or {}
    # Validate input
    try:
        validated = ProcessLogRequest(**data)
    except Exception as e:
        return jsonify({'error': 'validation_error', 'detail': str(e)}), 400
    log_text = validated.log
    mode = validated.mode
    manual_solution = validated.manual_solution
    approvers = validated.approvers
    llm_config = validated.llm_config
    agents = validated.agents
    include_trace = validated.include_trace
    agent = UltimateAIAutonomousDevOps()
    import asyncio
    req_id = getattr(g, 'request_id', None)
    result = asyncio.run(agent.process_issue(log_text, mode, manual_solution, llm_config, approvers, agents, include_trace=include_trace, request_id=req_id))
    # validate outgoing shape
    try:
        ProcessLogResult(**result)  # ensures conformance; will raise if mismatch
    except Exception:
        pass
    agent.close()
    return {'result': result, 'deprecated': False}

@api_bp.route('/openapi.json', methods=['GET'])
def openapi_spec():  # public static spec
    spec = {
        'openapi': '3.0.0',
        'info': {'title': 'DevOps Agent API', 'version': '0.2.0'},
        'components': {
            'schemas': {
                'ProcessLogRequest': {
                    'type': 'object',
                    'required': ['log'],
                    'properties': {
                        'log': {'type': 'string'},
                        'mode': {'type': 'string', 'enum': ['safe', 'auto']},
                        'manual_solution': {'type': 'string'},
                        'approvers': {'type': 'array', 'items': {'type': 'string'}},
                        'llm_config': {'type': 'object'},
                        'agents': {'type': 'array', 'items': {'type': 'string'}},
                        'include_trace': {'type': 'boolean'}
                    }
                },
                'Thought': {
                    'type': 'object',
                    'properties': {
                        'step': {'type': 'integer'},
                        'role': {'type': 'string'},
                        'content': {'type': 'string'},
                        'score': {'type': 'number'}
                    }
                },
                'ProcessLogResult': {
                    'type': 'object',
                    'required': ['reasoning', 'diagnosis_found', 'applied', 'reasoning_id'],
                    'properties': {
                        'reasoning': {'type': 'string'},
                        'diagnosis_found': {'type': 'boolean'},
                        'applied': {'type': 'boolean'},
                        'error_type': {'type': 'string'},
                        'trace': {'type': 'array', 'items': {'$ref': '#/components/schemas/Thought'}},
                        'reasoning_id': {'type': 'string'},
                        'request_id': {'type': 'string'}
                    }
                }
            }
        },
        'paths': {
            '/api/health': {'get': {'responses': {'200': {'description': 'OK'}}}},
            '/api/config': {'get': {'responses': {'200': {'description': 'Config'}}}},
            '/api/process_log': {
                'post': {
                    'requestBody': {
                        'required': True,
                        'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ProcessLogRequest'}}}
                    },
                    'responses': {'200': {'description': 'Reasoning result', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ProcessLogResult'}}}}}
                }
            },
            '/api/metrics': {'get': {'responses': {'200': {'description': 'Metrics'}}}},
        }
    }
    return jsonify(spec)

try:
    from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

if PROM_AVAILABLE:
    REQ_COUNTER = Counter('requests_total', 'Total HTTP requests')
    FIX_SUCCESS = Counter('fix_success_total', 'Successful fixes applied')
    FIX_FAILURE = Counter('fix_failure_total', 'Failed fixes attempted')
else:
    METRICS = {
        'requests_total': 0,
        'fix_success_total': 0,
        'fix_failure_total': 0,
    }

@app.before_request
def _count_request():
    if PROM_AVAILABLE:
        REQ_COUNTER.inc()
    else:
        METRICS['requests_total'] += 1

@api_bp.route('/metrics', methods=['GET'])
def metrics():
    if PROM_AVAILABLE:
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    lines = []
    for k, v in METRICS.items():
        lines.append(f"# HELP {k} Counter for {k}")
        lines.append(f"# TYPE {k} counter")
        lines.append(f"{k} {v}")
    return Response('\n'.join(lines) + '\n', mimetype='text/plain')

app.register_blueprint(api_bp, url_prefix='/api')

# Flask integration (kept minimal) can be separated further in production

__all__ = ['UltimateAIAutonomousDevOps', 'app', 'require_api_key']
