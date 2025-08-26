"""Lightweight cognitive scaffolding for iterative reasoning.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.

NOT real AGI. Provides a structured loop:
1. Planning
2. Execution
3. Critique
4. Memory write (future extension)

Upgradable: replace toolbelt callables with LLM / vector store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import time


@dataclass
class Thought:
    step: int
    role: str  # planner|executor|critic
    content: str
    score: float | None = None


@dataclass
class Task:
    description: str
    attempts: int = 0
    result: Optional[str] = None
    success: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


class Planner:
    def plan(self, log_excerpt: str) -> List[Task]:
        phases = [
            ("Classify error type", "classify"),
            ("Propose candidate fix", "propose_fix"),
            ("Derive revert logic", "revert"),
            ("Assess risk", "risk"),
        ]
        tasks = [Task(desc, metadata={"kind": kind}) for desc, kind in phases]
        if 'unknown' in log_excerpt.lower():
            tasks.append(Task("Search memory for similar patterns", metadata={"kind": "memory_lookup"}))
        return tasks


class Executor:
    def __init__(self, toolbelt: Dict[str, Callable]):
        self.toolbelt = toolbelt

    def execute(self, task: Task, context: Dict) -> Task:
        kind = task.metadata.get('kind')
        fn = self.toolbelt.get(kind)
        if fn:
            try:
                task.result = fn(context)
                task.success = True
            except Exception as e:
                task.result = f"error:{e}"
                task.success = False
        else:
            task.result = "noop"
            task.success = True
        task.attempts += 1
        return task


class Critic:
    def critique(self, tasks: List[Task]) -> float:
        if not tasks:
            return 0.0
        success = sum(1 for t in tasks if t.success)
        return success / len(tasks)

    def refinement_needed(self, score: float, threshold: float) -> bool:
        return score < threshold


class CognitiveLoop:
    def __init__(self, planner: Planner, executor: Executor, critic: Critic, max_cycles: int = 2, target_score: float = 0.75):
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.max_cycles = max_cycles
        self.target_score = target_score
        self.thoughts: List[Thought] = []

    def run(self, log_excerpt: str, context: Dict) -> Dict[str, str]:
        aggregate: Dict[str, str] = {}
        for cycle in range(1, self.max_cycles + 1):
            tasks = self.planner.plan(log_excerpt)
            self.thoughts.append(Thought(cycle, 'planner', f"Planned {len(tasks)} tasks"))
            for t in tasks:
                executed = self.executor.execute(t, context)
                if executed.metadata.get('kind') == 'propose_fix' and executed.result and 'fix:' in executed.result:
                    aggregate['suggested_fix'] = executed.result.split('fix:', 1)[1].strip()
                if executed.metadata.get('kind') == 'revert' and executed.result and 'revert:' in executed.result:
                    aggregate['revert_logic'] = executed.result.split('revert:', 1)[1].strip()
                if executed.metadata.get('kind') == 'classify' and executed.result and 'type:' in executed.result:
                    aggregate['error_type'] = executed.result.split('type:', 1)[1].strip()
                if executed.metadata.get('kind') == 'memory_lookup' and executed.result and executed.result.startswith('memory:'):
                    aggregate.setdefault('memory_refs', executed.result)
            score = self.critic.critique(tasks)
            self.thoughts.append(Thought(cycle, 'critic', f"score={score:.2f}", score=score))
            if not self.critic.refinement_needed(score, self.target_score):
                break
            context['refine_hint'] = f"cycle{cycle}_improve"
            time.sleep(0.005)
        return aggregate

__all__ = ['Thought', 'Task', 'Planner', 'Executor', 'Critic', 'CognitiveLoop']
