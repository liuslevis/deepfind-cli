from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentResult:
    name: str
    text: str
    citations: list[str] = field(default_factory=list)
    iterations: int = 0


@dataclass
class WorkerReport:
    task: str
    text: str
    citations: list[str]
    parsed: dict
