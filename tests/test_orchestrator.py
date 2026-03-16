from __future__ import annotations

import unittest
from unittest.mock import patch

from deepfind.config import Settings
from deepfind.orchestrator import DeepFind, WORKER_PROMPT


class OrchestratorTests(unittest.TestCase):
    def test_plan_pads_missing_tasks(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = '["one task"]'
            tasks = app._plan("topic", num_agent=3, max_iter=2)
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0], "one task")

    def test_worker_prompt_mentions_bili_transcribe(self) -> None:
        self.assertIn("bili_transcribe", WORKER_PROMPT)
        self.assertIn("Bilibili", WORKER_PROMPT)
