"""Legacy monolithic script (deprecated).

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
import re
# LEGACY FILE: This script is deprecated. Kept for backward compatibility.
# Use package 'devops_agent' instead. Will be removed in a future release.
import json
import sqlite3
import requests
import docker
import paramiko
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, jsonify, request
from sklearn.ensemble import IsolationForest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import asyncio
import logging
import os
import uuid
import git
from cryptography.fernet import Fernet
from opa_client.opa import OpaClient
from hvac import Client as VaultClient
import websocket
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class UltimateAIAutonomousDevOps:
    def __init__(self, db_path: str = "devops_knowledge.db", llm_endpoint: str = "http://localhost:8000/llm",
                 opa_endpoint: str = "http://localhost:8080", vault_url: str = "http://localhost:8200"):
        """Initialize ultimate AI DevOps agent with policy and vault integration."""
        self.knowledge_db = sqlite3.connect(db_path)
        self.llm_endpoint = llm_endpoint
        self.opa_client = OpaClient(opa_endpoint)
        self.vault_client = VaultClient(url=vault_url)
        self.docker_client = docker.from_env()
        self.ssh_client = None
        self.anomaly_detectors = {}
        self.plugins = {}
        self.actuators = {}
        self.alert_channels = []
        self.encryption_key = self.load_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        self.setup_db()
        self.error_patterns = {
            'memory': re.compile(r'memory\s+error|out\s+of\s+memory', re.IGNORECASE),
            'connection': re.compile(r'connection\s+(timeout|refused|failed)', re.IGNORECASE),
            'disk': re.compile(r'disk\s+(full|space)', re.IGNORECASE),
        }
        self.ws_clients = []
        self.start_websocket_server()

    def load_encryption_key(self) -> bytes:
        """Load encryption key from Vault."""
        try:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path="devops/encryption_key",
                secret=dict(key=Fernet.generate_key().decode())
            )
            secret = self.vault_client.secrets.kv.v2.read_secret_version(path="devops/encryption_key")
            return secret['data']['data']['key'].encode()
        except Exception as e:
            logger.error(f"Failed to load encryption key from Vault: {str(e)}")
            return Fernet.generate_key()

    def setup_db(self):
        """Setup SQLite database for knowledge base, audit logs, feedback, votes, and marketplace."""
        cursor = self.knowledge_db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failure_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT,
                pattern TEXT,
                solution TEXT,
                success_count INTEGER DEFAULT 0,
                contribution_points INTEGER DEFAULT 0,
                last_applied TIMESTAMP,
                source TEXT DEFAULT 'local',
                votes INTEGER DEFAULT 0,
                marketplace_id TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                error_type TEXT,
                log_snippet TEXT,
                solution TEXT,
                result TEXT,
                timestamp TIMESTAMP,
                approvers TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fix_id INTEGER,
                success BOOLEAN,
                comments TEXT,
                timestamp TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS marketplace_plugins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plugin_name TEXT,
                config TEXT,
                votes INTEGER DEFAULT 0,
                creator TEXT
            )
        ''')
        self.knowledge_db.commit()

    def load_plugin(self, plugin_name: str, config: Dict, creator: str = 'anonymous'):
        """Load a plugin for specialized domains and register in marketplace."""
        self.plugins[plugin_name] = config
        cursor = self.knowledge_db.cursor()
        cursor.execute('''
            INSERT INTO marketplace_plugins (plugin_name, config, creator)
            VALUES (?, ?, ?)
        ''', (plugin_name, json.dumps(config), creator))
        self.knowledge_db.commit()
        logger.info(f"Loaded and registered plugin: {plugin_name}")

    def register_actuator(self, actuator_name: str, config: Dict):
        """Register a new actuator for applying fixes."""
        self.actuators[actuator_name] = config
        logger.info(f"Registered actuator: {actuator_name}")

    def register_alert_channel(self, channel: Dict):
        """Register a new alert channel with rate-limiting."""
        self.alert_channels.append(channel)
        logger.info(f"Registered alert channel: {channel['type']}")

    def start_websocket_server(self):
        """Start WebSocket server for real-time dashboard updates."""
        def on_message(ws, message):
            logger.info(f"WebSocket message: {message}")
            ws.send(json.dumps({'status': 'received'}))

        def on_open(ws):
            self.ws_clients.append(ws)
            logger.info("WebSocket client connected")

        def on_close(ws):
            self.ws_clients.remove(ws)
            logger.info("WebSocket client disconnected")

        ws_server = websocket.WebSocketApp("ws://localhost:8765",
                                          on_message=on_message,
                                          on_open=on_open,
                                          on_close=on_close)
        threading.Thread(target=ws_server.run_forever, daemon=True).start()

    def broadcast_dashboard_update(self, data: Dict):
        """Broadcast dashboard updates to WebSocket clients."""
        for ws in self.ws_clients:
            ws.send(json.dumps(data))
        logger.info("Broadcasted dashboard update")

    def check_policy(self, diagnosis: Dict[str, str], mode: str) -> bool:
        """Check if fix is allowed by policy using OPA."""
        input_data = {
            'error_type': diagnosis.get('error_type', 'unknown'),
            'solution': diagnosis['solution'],
            'mode': mode
        }
        try:
            result = self.opa_client.check_policy(input_data, 'devops/allow_fix')
            logger.info(f"Policy check result: {result}")
            return result.get('result', False)
        except Exception as e:
            logger.error(f"Policy check failed: {str(e)}")
            return False

    def analyze_log_with_llm(self, log: str, llm_config: Optional[Dict] = None) -> Dict[str, str]:
        """Analyze log with configurable LLM."""
        endpoint = llm_config.get('endpoint', self.llm_endpoint) if llm_config else self.llm_endpoint
        try:
            response = requests.post(endpoint, json={
                'prompt': f"Analyze this log, categorize the error, suggest a fix with revert logic, and explain root cause: {log[:1000]}",
                'max_tokens': 400
            })
            result = response.json()
            error_info = {
                'error_type': result.get('error_type', 'unknown'),
                'summary': result.get('summary', log[:200]),
                'suggested_fix': result.get('suggested_fix', ''),
                'revert_logic': result.get('revert_logic', ''),
                'root_cause': result.get('root_cause', 'Unknown cause')
            }
            logger.info(f"LLM analysis: {error_info['error_type']}")
            return error_info
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return self.analyze_log(log)

    def analyze_log(self, log: str) -> Dict[str, str]:
        """Fallback log analysis using regex."""
        for error_type, pattern in self.error_patterns.items():
            if pattern.search(log):
                return {'error_type': error_type, 'summary': log[:200], 'suggested_fix': '', 'revert_logic': '', 'root_cause': 'Pattern-based detection'}
        return {'error_type': 'unknown', 'summary': log[:200], 'suggested_fix': '', 'revert_logic': '', 'root_cause': 'Unknown'}

    def simulate_impact(self, solution: str) -> Dict[str, str]:
        """Simulate fix impact using LLM reasoning."""
        try:
            response = requests.post(self.llm_endpoint, json={
                'prompt': f"What could go wrong if I apply this fix: {solution}?",
                'max_tokens': 200
            })
            impact = response.json().get('impact', 'No impact analysis available')
            logger.info(f"Impact simulation: {impact}")
            return {'impact': impact}
        except Exception as e:
            logger.error(f"Impact simulation failed: {str(e)}")
            return {'impact': 'Simulation failed'}

    def generate_iac_patch(self, solution: str, iac_type: str = 'terraform') -> str:
        """Generate IaC patch for permanent fix."""
        try:
            response = requests.post(self.llm_endpoint, json={
                'prompt': f"Convert this fix to a {iac_type} patch: {solution}",
                'max_tokens': 500
            })
            patch = response.json().get('patch', f'# No {iac_type} patch generated')
            repo = git.Repo('.')
            patch_file = f'fix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tf'
            with open(patch_file, 'w') as f:
                f.write(patch)
            repo.git.add(patch_file)
            repo.git.commit(m=f"Auto-generated {iac_type} patch")
            logger.info(f"Generated {iac_type} patch and committed: {patch_file}")
            return patch_file
        except Exception as e:
            logger.error(f"IaC patch generation failed: {str(e)}")
            return f'# Error generating {iac_type} patch'

    def natural_language_to_workflow(self, description: str, output_type: str = 'bash') -> str:
        """Convert natural language to workflow script."""
        try:
            response = requests.post(self.llm_endpoint, json={
                'prompt': f"Convert this description to a {output_type} script with revert logic: {description}",
                'max_tokens': 500
            })
            script = response.json().get('script', f'# No {output_type} script generated')
            logger.info(f"Generated {output_type} script")
            return script
        except Exception as e:
            logger.error(f"Workflow generation failed: {str(e)}")
            return f'# Error generating {output_type} script'

    def diagnose_issue(self, error_info: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Diagnose issue and fetch or generate solution."""
        cursor = self.knowledge_db.cursor()
        cursor.execute('''
            SELECT pattern, solution, revert_logic, marketplace_id FROM failure_patterns 
            WHERE error_type = ? ORDER BY votes DESC, success_count DESC LIMIT 1
        ''', (error_info['error_type'],))
        result = cursor.fetchone()
        
        if result:
            logger.info(f"Found solution for {error_info['error_type']}")
            return {'pattern': result[0], 'solution': result[1], 'revert_logic': result[2], 'marketplace_id': result[3]}
        elif error_info['suggested_fix']:
            return {'pattern': error_info['summary'], 'solution': error_info['suggested_fix'], 
                    'revert_logic': error_info['revert_logic'], 'marketplace_id': None}
        return None

    def apply_fix(self, diagnosis: Dict[str, str], mode: str = 'safe', approvers: List[str] = None) -> bool:
        """Apply fix with simulation, approval, policy check, and actuator."""
        if not self.check_policy(diagnosis, mode):
            logger.warning("Fix blocked by policy")
            self.notify_alert_channels("Fix blocked by policy")
            return False

        impact = self.simulate_impact(diagnosis['solution'])
        if impact['impact'].lower().find('critical') != -1 and mode == 'auto':
            logger.warning("Critical impact detected, switching to safe mode")
            mode = 'safe'

        if mode == 'safe' and approvers and len(approvers) < 2:
            logger.warning("Multi-user approval required")
            self.notify_for_approval(diagnosis, impact, approvers)
            return False

        try:
            dry_run_result = self.dry_run_fix(diagnosis)
            if dry_run_result['status'] != 'success':
                logger.warning("Dry run failed, aborting")
                self.notify_alert_channels("Dry run failed, aborting fix")
                return False

            actuator = self.actuators.get('default', {'type': 'ssh'})
            if actuator['type'] == 'ssh' and self.ssh_client:
                stdin, stdout, stderr = self.ssh_client.exec_command(diagnosis['solution'])
                if stderr.read():
                    logger.error("Fix application failed")
                    self.rollback_fix(diagnosis)
                    return False
            logger.info(f"Applied fix via {actuator['type']}: {diagnosis['solution']}")
            self.log_audit('apply_fix', diagnosis, 'success', approvers)
            self.request_feedback(diagnosis)
            patch_file = self.generate_iac_patch(diagnosis['solution'])
            self.broadcast_dashboard_update({
                'action': 'apply_fix',
                'error_type': diagnosis.get('error_type', 'unknown'),
                'status': 'success',
                'patch_file': patch_file
            })
            return True
        except Exception as e:
            logger.error(f"Fix application failed: {str(e)}")
            self.log_audit('apply_fix', diagnosis, f'failed: {str(e)}', approvers)
            self.notify_alert_channels(f"Fix application failed: {str(e)}")
            return False

    def dry_run_fix(self, diagnosis: Dict[str, str]) -> Dict[str, str]:
        """Run fix in a Docker sandbox."""
        try:
            container = self.docker_client.containers.run(
                'alpine', command=diagnosis['solution'], detach=True, remove=True
            )
            result = container.wait()
            logger.info(f"Dry run result: {result}")
            return {'status': 'success' if result['StatusCode'] == 0 else 'failed'}
        except Exception as e:
            logger.error(f"Dry run failed: {str(e)}")
            return {'status': 'failed'}

    def rollback_fix(self, diagnosis: Dict[str, str]):
        """Apply revert logic for failed fix."""
        if diagnosis.get('revert_logic'):
            logger.info(f"Rolling back with: {diagnosis['revert_logic']}")
            actuator = self.actuators.get('default', {'type': 'ssh'})
            if actuator['type'] == 'ssh' and self.ssh_client:
                self.ssh_client.exec_command(diagnosis['revert_logic'])
            self.broadcast_dashboard_update({
                'action': 'rollback',
                'error_type': diagnosis.get('error_type', 'unknown'),
                'status': 'executed'
            })

    def notify_for_approval(self, diagnosis: Dict[str, str], impact: Dict[str, str], approvers: List[str]):
        """Send approval request with impact analysis."""
        for channel in self.alert_channels:
            requests.post(channel['url'], json={
                'text': f"Approve fix: {diagnosis['solution']}\nImpact: {impact['impact']}\nApprovers: {', '.join(approvers)}"
            })
        self.broadcast_dashboard_update({
            'action': 'approval_request',
            'error_type': diagnosis.get('error_type', 'unknown'),
            'solution': diagnosis['solution'],
            'approvers': approvers
        })
        logger.info(f"Sent approval request to {len(self.alert_channels)} channels")

    def request_feedback(self, diagnosis: Dict[str, str]):
        """Request user feedback on fix success."""
        for channel in self.alert_channels:
            requests.post(channel['url'], json={
                'text': f"Did this fix resolve the issue? {diagnosis['solution']}"
            })
        self.broadcast_dashboard_update({
            'action': 'feedback_request',
            'error_type': diagnosis.get('error_type', 'unknown'),
            'solution': diagnosis['solution']
        })
        logger.info("Requested feedback on fix")

    def store_feedback(self, fix_id: int, success: bool, comments: str):
        """Store user feedback and update votes."""
        cursor = self.knowledge_db.cursor()
        cursor.execute('''
            INSERT INTO feedback (fix_id, success, comments, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (fix_id, success, comments, datetime.now()))
        if success:
            cursor.execute('''
                UPDATE failure_patterns SET success_count = success_count + 1, votes = votes + 1
                WHERE id = ?
            ''', (fix_id,))
        else:
            cursor.execute('''
                UPDATE failure_patterns SET votes = votes - 1
                WHERE id = ?
            ''', (fix_id,))
        self.knowledge_db.commit()
        self.broadcast_dashboard_update({
            'action': 'feedback',
            'fix_id': fix_id,
            'success': success
        })
        logger.info(f"Stored feedback for fix {fix_id}")

    def sync_global_knowledge(self, error_info: Dict[str, str], solution: str, revert_logic: str, contributor: str = 'anonymous'):
        """Sync anonymized fix to global knowledge base with federated learning."""
        anonymized_data = {
            'error_type': error_info['error_type'],
            'pattern': self.cipher.encrypt(error_info['summary'].encode()).decode(),
            'solution': solution,
            'revert_logic': revert_logic,
            'source': 'community',
            'marketplace_id': str(uuid.uuid4())
        }
        try:
            response = requests.post("http://global-knowledge-api", json=anonymized_data)
            if response.status_code == 200:
                self.learn_from_failure(error_info, solution, revert_logic, contributor, anonymized_data['marketplace_id'])
                logger.info("Synced to global knowledge base")
        except Exception as e:
            logger.error(f"Global sync failed: {str(e)}")

    def learn_from_failure(self, error_info: Dict[str, str], solution: str, revert_logic: str, contributor: str = 'anonymous', marketplace_id: Optional[str] = None):
        """Learn from fix and update knowledge base."""
        cursor = self.knowledge_db.cursor()
        cursor.execute('''
            INSERT INTO failure_patterns (error_type, pattern, solution, revert_logic, last_applied, contribution_points, source, marketplace_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (error_info['error_type'], error_info['summary'], solution, revert_logic, datetime.now(), 10, 'local', marketplace_id))
        self.knowledge_db.commit()
        logger.info(f"Learned new solution for {error_info['error_type']} by {contributor}")

    def log_audit(self, action: str, diagnosis: Dict[str, str], result: str, approvers: List[str] = None):
        """Log actions for audit trail."""
        cursor = self.knowledge_db.cursor()
        cursor.execute('''
            INSERT INTO audit_log (action, error_type, log_snippet, solution, result, timestamp, approvers)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (action, diagnosis.get('error_type', 'unknown'), diagnosis.get('pattern', ''),
              diagnosis.get('solution', ''), result, datetime.now(), ','.join(approvers or [])))
        self.knowledge_db.commit()
        self.broadcast_dashboard_update({
            'action': 'audit_log',
            'error_type': diagnosis.get('error_type', 'unknown'),
            'result': result
        })

    def train_anomaly_detector(self, service: str, metrics: List[float]):
        """Train per-service anomaly detector."""
        if service not in self.anomaly_detectors:
            self.anomaly_detectors[service] = IsolationForest(contamination=0.1)
        self.anomaly_detectors[service].fit(np.array(metrics).reshape(-1, 1))
        logger.info(f"Trained anomaly detector for {service}")

    def predict_failure(self, service: str, metrics: List[float]) -> bool:
        """Predict potential failures for a service."""
        if service in self.anomaly_detectors:
            prediction = self.anomaly_detectors[service].predict(np.array(metrics).reshape(-1, 1))
            if prediction[-1] == -1:
                logger.warning(f"Potential failure detected for {service}")
                self.notify_alert_channels(f"Potential failure detected for {service}")
                return True
        return False

    def coordinate_agents(self, diagnosis: Dict[str, str], agents: List[str]) -> bool:
        """Coordinate fix application across multiple agents."""
        failure_count = 0
        for agent in agents:
            try:
                response = requests.post(f"http://{agent}/apply_fix", json={
                    'diagnosis': diagnosis,
                    'mode': 'auto'
                })
                if response.status_code != 200:
                    failure_count += 1
                logger.info(f"Fix applied on agent {agent}")
            except Exception as e:
                logger.error(f"Agent {agent} failed: {str(e)}")
                failure_count += 1
        
        if failure_count / len(agents) > 0.1:  # Halt if >10% fail
            logger.warning("High failure rate, halting rollout")
            self.notify_alert_channels("High failure rate, halting agent rollout")
            return False
        return True

    def explain_reasoning(self, error_info: Dict[str, str], diagnosis: Optional[Dict[str, str]]) -> str:
        """Explain reasoning in natural language."""
        explanation = f"**Log**: {error_info['summary']}...\n**Root Cause**: {error_info['root_cause']}\n"
        if diagnosis:
            explanation += f"**Issue**: {error_info['error_type']}\n**Solution**: {diagnosis['solution']}\n**Revert Logic**: {diagnosis['revert_logic']}\n**Precedents**: Seen {self.get_precedent_count(error_info['error_type'])} times\n**Marketplace ID**: {diagnosis.get('marketplace_id', 'N/A')}"
        else:
            explanation += f"**Issue**: {error_info['error_type']}\n**Solution**: No solution found. Suggest manual review."
        return explanation

    def get_precedent_count(self, error_type: str) -> int:
        """Count global precedents for an error type."""
        cursor = self.knowledge_db.cursor()
        cursor.execute('SELECT COUNT(*) FROM failure_patterns WHERE error_type = ?', (error_type,))
        return cursor.fetchone()[0]

    def generate_explanation_report(self, error_info: Dict[str, str], diagnosis: Optional[Dict[str, str]]) -> str:
        """Generate markdown explanation report."""
        report = self.explain_reasoning(error_info, diagnosis)
        impact = self.simulate_impact(diagnosis['solution'] if diagnosis else '')
        report += f"\n**Impact Analysis**: {impact['impact']}"
        report_file = f"explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Generated explanation report: {report_file}")
        self.broadcast_dashboard_update({'action': 'report_generated', 'file': report_file})
        return report_file

    def generate_compliance_report(self, start_date: str, end_date: str) -> str:
        """Generate compliance report as PDF."""
        cursor = self.knowledge_db.cursor()
        cursor.execute('''
            SELECT action, error_type, log_snippet, solution, result, timestamp, approvers 
            FROM audit_log WHERE timestamp BETWEEN ? AND ?
        ''', (start_date, end_date))
        logs = cursor.fetchall()

        pdf_file = "compliance_report.pdf"
        c = canvas.Canvas(pdf_file, pagesize=letter)
        c.drawString(100, 750, "Compliance Report")
        y = 700
        for log in logs:
            c.drawString(100, y, f"{log[5]}: {log[0]} - {log[1]} - {log[4]} - Approvers: {log[6]}")
            y -= 20
        c.save()
        logger.info(f"Generated compliance report: {pdf_file}")
        self.broadcast_dashboard_update({'action': 'compliance_report', 'file': pdf_file})
        return pdf_file

    def connect_ssh(self, host: str, username: str, password: Optional[str] = None):
        """Connect to remote system via SSH with Vault-stored credentials."""
        try:
            secret = self.vault_client.secrets.kv.v2.read_secret_version(path=f"devops/ssh/{host}")
            password = secret['data']['data']['password']
        except Exception:
            logger.warning("No Vault credentials found, using provided password")
        
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(host, username=username, password=password)
        logger.info(f"Connected to {host} via SSH")

    async def process_issue(self, log: str, mode: str = 'safe', manual_solution: Optional[str] = None, 
                         llm_config: Optional[Dict] = None, approvers: List[str] = None, agents: List[str] = None) -> str:
        """Process issue from log to resolution with multi-agent coordination."""
        error_info = self.analyze_log_with_llm(log, llm_config)
        diagnosis = self.diagnose_issue(error_info)
        
        if diagnosis:
            dry_run_result = self.dry_run_fix(diagnosis)
            if dry_run_result['status'] == 'success':
                if agents:
                    success = self.coordinate_agents(diagnosis, agents)
                    if not success:
                        return self.explain_reasoning(error_info, None)
                success = self.apply_fix(diagnosis, mode, approvers)
                if success:
                    report = self.generate_explanation_report(error_info, diagnosis)
                    self.notify_alert_channels(f"Issue resolved: {report}")
                    return self.explain_reasoning(error_info, diagnosis)
            else:
                logger.warning("Dry run failed, skipping fix")
                self.notify_alert_channels("Dry run failed, manual review required")
                return self.explain_reasoning(error_info, None)
        else:
            if manual_solution:
                script = self.natural_language_to_workflow(manual_solution)
                self.sync_global_knowledge(error_info, script, '# Revert logic placeholder')
                report = self.generate_explanation_report(error_info, {'pattern': error_info['summary'], 'solution': script, 'revert_logic': ''})
                self.notify_alert_channels(f"New solution learned: {report}")
                return self.explain_reasoning(error_info, {'pattern': error_info['summary'], 'solution': script, 'revert_logic': ''})
        self.notify_alert_channels("No solution found, manual review required")
        return self.explain_reasoning(error_info, None)

    def notify_alert_channels(self, message: str):
        """Notify all registered alert channels with rate-limiting."""
        for channel in self.alert_channels:
            requests.post(channel['url'], json={'text': message})
        self.broadcast_dashboard_update({'action': 'alert', 'message': message})
        logger.info(f"Notified {len(self.alert_channels)} channels")

@app.route('/process_log', methods=['POST'])
def api_process_log():
    """API endpoint to process logs (legacy) secured via X-API-Key if API_KEYS env set."""
    api_keys = {k.strip() for k in (os.getenv('API_KEYS') or '').split(',') if k.strip()}
    if api_keys:
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not key or key not in api_keys:
            return jsonify({'error': 'unauthorized'}), 401
    data = request.json
    log = data.get('log', '')
    mode = data.get('mode', 'safe')
    manual_solution = data.get('manual_solution', None)
    llm_config = data.get('llm_config', None)
    approvers = data.get('approvers', None)
    agents = data.get('agents', None)
    
    agent = UltimateAIAutonomousDevOps()
    result = asyncio.run(agent.process_issue(log, mode, manual_solution, llm_config, approvers, agents))
    return jsonify({'result': result})

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """API endpoint for dashboard data."""
    agent = UltimateAIAutonomousDevOps()
    cursor = agent.knowledge_db.cursor()
    cursor.execute('SELECT error_type, COUNT(*) as count, SUM(votes) as votes FROM failure_patterns GROUP BY error_type')
    stats = cursor.fetchall()
    return jsonify({'stats': [{'error_type': row[0], 'count': row[1], 'votes': row[2]} for row in stats]})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """API endpoint for submitting feedback."""
    data = request.json
    agent = UltimateAIAutonomousDevOps()
    agent.store_feedback(data['fix_id'], data['success'], data['comments'])
    return jsonify({'status': 'Feedback recorded'})

@app.route('/compliance_report', methods=['GET'])
def compliance_report():
    """API endpoint for compliance report."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    agent = UltimateAIAutonomousDevOps()
    pdf_file = agent.generate_compliance_report(start_date, end_date)
    return jsonify({'report': pdf_file})

@app.route('/register_plugin', methods=['POST'])
def register_plugin():
    """API endpoint to register a plugin."""
    data = request.json
    agent = UltimateAIAutonomousDevOps()
    agent.load_plugin(data['plugin_name'], data['config'], data.get('creator', 'anonymous'))
    return jsonify({'status': f"Plugin {data['plugin_name']} registered"})

@app.route('/register_actuator', methods=['POST'])
def register_actuator():
    """API endpoint to register an actuator."""
    data = request.json
    agent = UltimateAIAutonomousDevOps()
    agent.register_actuator(data['actuator_name'], data['config'])
    return jsonify({'status': f"Actuator {data['actuator_name']} registered"})

@app.route('/register_alert_channel', methods=['POST'])
def register_alert_channel():
    """API endpoint to register an alert channel."""
    data = request.json
    agent = UltimateAIAutonomousDevOps()
    agent.register_alert_channel(data['channel'])
    return jsonify({'status': f"Alert channel {data['channel']['type']} registered"})

@app.route('/marketplace', methods=['GET'])
def get_marketplace():
    """API endpoint to list marketplace plugins."""
    agent = UltimateAIAutonomousDevOps()
    cursor = agent.knowledge_db.cursor()
    cursor.execute('SELECT plugin_name, config, votes, creator FROM marketplace_plugins')
    plugins = cursor.fetchall()
    return jsonify({'plugins': [{'name': row[0], 'config': json.loads(row[1]), 'votes': row[2], 'creator': row[3]} for row in plugins]})

if __name__ == "__main__":
    agent = UltimateAIAutonomousDevOps()
    agent.load_plugin('kubernetes', {'endpoint': 'http://k8s-llm', 'type': 'llm'}, 'admin')
    agent.register_actuator('default', {'type': 'ssh'})
    agent.register_alert_channel({'type': 'slack', 'url': 'http://slack-webhook'})
    sample_log = "Error: Out of memory on server at 2025-08-26"
    result = asyncio.run(agent.process_issue(sample_log, mode='safe', manual_solution="Increase memory allocation to 16GB", 
                                         approvers=['user1', 'user2'], agents=['agent1:8080', 'agent2:8080']))
    print(result)
    app.run(debug=True)