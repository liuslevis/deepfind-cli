from __future__ import annotations

import unittest
from unittest.mock import patch

from deepfind.config import Settings
from deepfind.models import ChatMessage, WorkerReport
from deepfind.orchestrator import (
    DeepFind,
    LEAD_PROMPT,
    PLAN_PROMPT,
    SYNTHESIS_PROMPT,
    WORKER_PROMPT,
)


class OrchestratorTests(unittest.TestCase):
    def test_plan_pads_missing_tasks(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = '["one task"]'
            tasks = app._plan("topic", transcript=[], num_agent=3, max_iter=2)
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0], "one task")
        self.assertTrue(agent_cls.return_value.run.call_args.kwargs["use_tools"])

    def test_worker_prompt_mentions_bilibili_tools(self) -> None:
        self.assertIn("boss_search", WORKER_PROMPT)
        self.assertIn("bili_search", WORKER_PROMPT)
        self.assertIn("bili_get_user_videos", WORKER_PROMPT)
        self.assertIn("bili_transcribe", WORKER_PROMPT)
        self.assertIn("bili_transcribe_full", WORKER_PROMPT)
        self.assertIn("short query", WORKER_PROMPT)
        self.assertIn("youtube_transcribe", WORKER_PROMPT)
        self.assertIn("web_fetch", WORKER_PROMPT)
        self.assertIn("Bilibili", WORKER_PROMPT)
        self.assertIn("BOSS Zhipin", WORKER_PROMPT)
        self.assertIn("YouTube", WORKER_PROMPT)
        self.assertIn("web_search", WORKER_PROMPT)
        self.assertIn("gen_img", WORKER_PROMPT)
        self.assertIn("gen_slides", WORKER_PROMPT)

    def test_synthesis_prompt_mentions_web_fetch(self) -> None:
        self.assertIn("web_search", SYNTHESIS_PROMPT)
        self.assertIn("web_fetch", SYNTHESIS_PROMPT)
        self.assertIn("JSON only", SYNTHESIS_PROMPT)

    def test_lead_prompt_mentions_gen_img(self) -> None:
        self.assertIn("gen_img", LEAD_PROMPT)
        self.assertIn("gen_slides", LEAD_PROMPT)
        self.assertIn("synthesis", LEAD_PROMPT)

    def test_plan_prompt_mentions_slides(self) -> None:
        self.assertIn("slides", PLAN_PROMPT)
        self.assertIn("web_search", PLAN_PROMPT)
        self.assertIn("web_fetch", PLAN_PROMPT)
        self.assertIn("BOSS Zhipin", PLAN_PROMPT)
        self.assertIn("YouTube", PLAN_PROMPT)

    def test_synthesize_falls_back_when_output_is_not_json(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        reports = [
            WorkerReport(
                task="task",
                text="report text",
                citations=["https://example.com/source"],
                parsed={
                    "summary": "worker summary",
                    "facts": [{"point": "fact", "source": "https://example.com/source"}],
                    "gaps": ["missing_data"],
                },
            )
        ]
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = "not json"
            synthesis = app._synthesize("topic", transcript=[], reports=reports, max_iter=2)
        self.assertEqual(synthesis["summary"], "worker summary")
        self.assertEqual(synthesis["sources"], ["https://example.com/source"])
        self.assertIn("Investigate gap: missing_data", synthesis["next_steps"])

    def test_run_turn_detailed_passes_synthesis_to_lead(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        fake_reports = []
        fake_synthesis = {"summary": "syn", "evidence": [], "gaps": [], "sources": [], "next_steps": []}
        with patch.object(app, "_plan", return_value=["task"]) as plan:
            with patch.object(app, "_run_workers", return_value=fake_reports) as run_workers:
                with patch.object(app, "_synthesize", return_value=fake_synthesis) as synthesize:
                    with patch.object(app, "_lead", return_value="final answer") as lead:
                        answer, reports = app._run_turn_detailed("topic", transcript=[], num_agent=1, max_iter_per_agent=2)
        self.assertEqual(answer, "final answer")
        self.assertEqual(reports, fake_reports)
        plan.assert_called_once()
        run_workers.assert_called_once()
        synthesize.assert_called_once()
        self.assertEqual(lead.call_args.args[2], fake_synthesis)

    def test_lead_uses_only_asset_tools_when_requested(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        synthesis = {"summary": "syn", "evidence": [], "gaps": [], "sources": [], "next_steps": []}
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = "answer"
            app._lead("Generate slides from this summary", transcript=[], synthesis=synthesis, max_iter=2)
        self.assertTrue(agent_cls.return_value.run.call_args.kwargs["use_tools"])
        self.assertEqual(agent_cls.return_value.run.call_args.kwargs["tool_names"], ["gen_slides"])

    def test_lead_skips_tools_for_normal_research_answer(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        synthesis = {"summary": "syn", "evidence": [], "gaps": [], "sources": [], "next_steps": []}
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = "answer"
            app._lead("Summarize the findings", transcript=[], synthesis=synthesis, max_iter=2)
        self.assertFalse(agent_cls.return_value.run.call_args.kwargs["use_tools"])
        self.assertIsNone(agent_cls.return_value.run.call_args.kwargs["tool_names"])

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
