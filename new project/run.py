# Copyright (c) 2025 Patrick Morrison
# Licensed under the MIT License. See LICENSE file for details.
from devops_agent.agent import UltimateAIAutonomousDevOps, app

# Simple factory so other modules (WSGI) can import create_app()
def create_app():
    return app

if __name__ == '__main__':
    with UltimateAIAutonomousDevOps() as agent:
        agent.load_plugin('kubernetes', {'endpoint': 'http://k8s-llm', 'type': 'llm'}, 'admin')
        agent.register_actuator('default', {'type': 'ssh'})
        agent.register_alert_channel({'type': 'slack', 'url': 'http://slack-webhook'})
        app.run(host='0.0.0.0', port=5000)
