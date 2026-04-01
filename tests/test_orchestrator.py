from __future__ import annotations

import unittest
from unittest.mock import patch

from deepfind.config import Settings
from deepfind.models import ChatMessage
from deepfind.orchestrator import DeepFind, LEAD_PROMPT, PLAN_PROMPT, WORKER_PROMPT


class OrchestratorTests(unittest.TestCase):
    def test_plan_pads_missing_tasks(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = '["one task"]'
            tasks = app._plan("topic", transcript=[], num_agent=3, max_iter=2)
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0], "one task")

    def test_worker_prompt_mentions_bili_transcribe(self) -> None:
        self.assertIn("boss_search", WORKER_PROMPT)
        self.assertIn("bili_transcribe", WORKER_PROMPT)
        self.assertIn("youtube_transcribe", WORKER_PROMPT)
        self.assertIn("Bilibili", WORKER_PROMPT)
        self.assertIn("BOSS Zhipin", WORKER_PROMPT)
        self.assertIn("YouTube", WORKER_PROMPT)
        self.assertIn("web_search", WORKER_PROMPT)
        self.assertIn("gen_img", WORKER_PROMPT)
        self.assertIn("gen_slides", WORKER_PROMPT)

    def test_lead_prompt_mentions_gen_img(self) -> None:
        self.assertIn("web_search", LEAD_PROMPT)
        self.assertIn("boss_chatlist", LEAD_PROMPT)
        self.assertIn("BOSS Zhipin", LEAD_PROMPT)
        self.assertIn("YouTube", LEAD_PROMPT)
        self.assertIn("gen_img", LEAD_PROMPT)
        self.assertIn("gen_slides", LEAD_PROMPT)

    def test_plan_prompt_mentions_slides(self) -> None:
        self.assertIn("slides", PLAN_PROMPT)
        self.assertIn("web_search", PLAN_PROMPT)
        self.assertIn("BOSS Zhipin", PLAN_PROMPT)
        self.assertIn("YouTube", PLAN_PROMPT)

    def test_chat_session_keeps_full_successful_transcript(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        session = app.session(num_agent=1, max_iter_per_agent=2)
        with patch.object(app, "_run_turn", side_effect=["first answer", "second answer"]) as run_turn:
            session.ask("first question")
            session.ask("follow up")
        self.assertEqual(run_turn.call_args_list[0].kwargs["transcript"], [])
        self.assertEqual(
            run_turn.call_args_list[1].kwargs["transcript"],
            [
                ChatMessage(role="user", content="first question"),
                ChatMessage(role="assistant", content="first answer"),
            ],
        )
