from __future__ import annotations

import unittest
from unittest.mock import patch

from deepfind.config import Settings
from deepfind.models import ChatMessage, WorkerReport
from deepfind.orchestrator import (
    DeepFind,
    FORMAT_FOLLOWUP_PROMPT,
    LEAD_PROMPT,
    LONG_REPORT_LEAD_PROMPT,
    PLAN_PROMPT,
    SYNTHESIS_PROMPT,
    WORKER_PROMPT,
    _LONG_REPORT_LEAD_MAX_TOKENS,
    _canonicalize_url,
    _parse_report,
    _should_shortcut_format_follow_up,
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

    def test_worker_prompt_mentions_claims_schema(self) -> None:
        self.assertIn("boss_search", WORKER_PROMPT)
        self.assertIn("bili_search", WORKER_PROMPT)
        self.assertIn("bili_transcribe", WORKER_PROMPT)
        self.assertIn("youtube_transcribe", WORKER_PROMPT)
        self.assertIn("youtube_transcribe_full", WORKER_PROMPT)
        self.assertNotIn("youtube_audio_transcribe", WORKER_PROMPT)
        self.assertIn('"claims"', WORKER_PROMPT)
        self.assertIn('"citations"', WORKER_PROMPT)
        self.assertIn('"confidence"', WORKER_PROMPT)

    def test_synthesis_prompt_mentions_key_points(self) -> None:
        self.assertIn("web_search", SYNTHESIS_PROMPT)
        self.assertIn("web_fetch", SYNTHESIS_PROMPT)
        self.assertIn('"overview_md"', SYNTHESIS_PROMPT)
        self.assertIn('"key_points"', SYNTHESIS_PROMPT)

    def test_lead_prompt_mentions_asset_tools(self) -> None:
        self.assertIn("gen_img", LEAD_PROMPT)
        self.assertIn("gen_slides", LEAD_PROMPT)
        self.assertIn("synthesis", LEAD_PROMPT)
        self.assertIn("[1]", LEAD_PROMPT)

    def test_long_report_prompt_mentions_required_sections(self) -> None:
        self.assertIn("## Conclusion", LONG_REPORT_LEAD_PROMPT)
        self.assertIn("## Reference", LONG_REPORT_LEAD_PROMPT)
        self.assertIn("current language", LONG_REPORT_LEAD_PROMPT)
        self.assertIn("[1]", LONG_REPORT_LEAD_PROMPT)

    def test_plan_prompt_mentions_slides(self) -> None:
        self.assertIn("slides", PLAN_PROMPT)
        self.assertIn("web_search", PLAN_PROMPT)
        self.assertIn("web_fetch", PLAN_PROMPT)
        self.assertIn("BOSS Zhipin", PLAN_PROMPT)
        self.assertIn("YouTube", PLAN_PROMPT)

    def test_parse_report_non_json_does_not_fabricate_citations(self) -> None:
        report = _parse_report("sub-1", "task", "plain text result", ["https://example.com/source"])

        self.assertEqual(report.agent_id, "sub-1")
        self.assertEqual(report.parsed["summary"], "plain text result")
        self.assertEqual(report.parsed["claims"], [])
        self.assertEqual(report.parsed["gaps"], ["non_json_output"])

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
                    "claims": [
                        {
                            "text": "fact",
                            "citations": ["https://example.com/source"],
                            "confidence": "high",
                        }
                    ],
                    "gaps": ["missing_data"],
                },
                agent_id="sub-1",
            )
        ]
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = "not json"
            synthesis = app._synthesize("topic", transcript=[], reports=reports, max_iter=2)
        self.assertEqual(synthesis["overview_md"], "worker summary")
        self.assertEqual(synthesis["key_points"][0]["text"], "fact")
        self.assertEqual(synthesis["key_points"][0]["citations"], ["https://example.com/source"])
        self.assertIn("Investigate gap: missing_data", synthesis["next_steps"])

    def test_run_turn_structured_builds_citation_occurrences_and_dedup(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        fake_reports = [
            WorkerReport(
                task="task a",
                text="worker a",
                citations=[],
                parsed={
                    "summary": "summary a",
                    "claims": [
                        {
                            "text": "claim a",
                            "citations": [
                                "https://EXAMPLE.com/report?utm_source=news#top",
                                "https://example.com/report",
                            ],
                            "confidence": "high",
                        }
                    ],
                    "gaps": [],
                },
                agent_id="sub-1",
            ),
            WorkerReport(
                task="task b",
                text="worker b",
                citations=[],
                parsed={
                    "summary": "summary b",
                    "claims": [
                        {
                            "text": "claim b",
                            "citations": ["https://example.com/report?utm_medium=social"],
                            "confidence": "medium",
                        }
                    ],
                    "gaps": ["missing"],
                },
                agent_id="sub-2",
            ),
        ]
        fake_synthesis = {
            "overview_md": "draft overview",
            "key_points": [
                {
                    "text": "lead point",
                    "citations": ["https://example.com/report?utm_campaign=launch"],
                    "confidence": "low",
                }
            ],
            "disagreements": ["workers differ on size"],
            "gaps": ["missing"],
            "next_steps": ["check primary source"],
        }
        with patch.object(app, "_plan", return_value=["task a", "task b"]) as plan:
            with patch.object(app, "_run_workers", return_value=fake_reports) as run_workers:
                with patch.object(app, "_synthesize", return_value=fake_synthesis) as synthesize:
                    with patch.object(app, "_lead", return_value="final lead overview") as lead:
                        envelope, reports = app._run_turn_structured(
                            "topic",
                            transcript=[],
                            num_agent=2,
                            max_iter_per_agent=2,
                        )

        self.assertEqual(reports, fake_reports)
        self.assertEqual(envelope["lead"]["overview_md"], "final lead overview")
        self.assertEqual(envelope["agents"][0]["claims"][0]["citation_ids"], ["c1"])
        self.assertEqual(envelope["agents"][1]["claims"][0]["citation_ids"], ["c1"])
        self.assertEqual(envelope["lead"]["key_points"][0]["citation_ids"], ["c1"])
        self.assertEqual(len(envelope["citations"]), 4)
        self.assertEqual(len(envelope["citations_dedup"]), 1)
        self.assertEqual(envelope["citations_dedup"][0]["canonical_url"], "https://example.com/report")
        self.assertEqual(
            [item["source_agent"] for item in envelope["citations"]],
            ["sub-1", "sub-1", "sub-2", "lead"],
        )
        self.assertEqual(
            [item["source_section"] for item in envelope["citations"]],
            ["claim", "claim", "claim", "key_point"],
        )
        self.assertEqual(envelope["lead"]["disagreements"], ["workers differ on size"])
        self.assertEqual(envelope["lead"]["next_steps"], ["check primary source"])
        self.assertEqual(envelope["meta"]["num_agents"], 2)
        self.assertEqual(envelope["meta"]["max_iter_per_agent"], 2)
        plan.assert_called_once()
        run_workers.assert_called_once()
        synthesize.assert_called_once()
        lead.assert_called_once()

    def test_run_turn_structured_passes_numbered_references_to_lead(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        fake_reports = [
            WorkerReport(
                task="task",
                text="worker",
                citations=[],
                parsed={
                    "summary": "summary",
                    "claims": [
                        {
                            "text": "claim",
                            "citations": ["https://example.com/source?utm_source=news"],
                            "confidence": "high",
                        }
                    ],
                    "gaps": [],
                },
                agent_id="sub-1",
            )
        ]
        fake_synthesis = {
            "overview_md": "draft overview",
            "key_points": [
                {
                    "text": "lead point",
                    "citations": ["https://example.com/source"],
                    "confidence": "medium",
                }
            ],
            "disagreements": [],
            "gaps": [],
            "next_steps": [],
        }
        with patch.object(app, "_plan", return_value=["task"]):
            with patch.object(app, "_run_workers", return_value=fake_reports):
                with patch.object(app, "_synthesize", return_value=fake_synthesis):
                    with patch.object(app, "_lead", return_value="final lead overview") as lead:
                        app._run_turn_structured(
                            "topic",
                            transcript=[],
                            num_agent=1,
                            max_iter_per_agent=2,
                        )

        lead_synthesis = lead.call_args.args[2]
        self.assertEqual(
            lead_synthesis["references"],
            [
                {
                    "number": 1,
                    "citation_id": "c1",
                    "url": "https://example.com/source",
                    "title": "",
                    "publisher": "",
                }
            ],
        )
        self.assertEqual(
            lead_synthesis["key_points"],
            [
                {
                    "text": "lead point",
                    "citation_ids": ["c1"],
                    "reference_numbers": [1],
                    "confidence": "medium",
                }
            ],
        )

    def test_lead_uses_only_asset_tools_when_requested(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        synthesis = {"overview_md": "syn", "key_points": [], "disagreements": [], "gaps": [], "next_steps": []}
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = "answer"
            app._lead("Generate slides from this summary", transcript=[], synthesis=synthesis, max_iter=2)
        self.assertTrue(agent_cls.return_value.run.call_args.kwargs["use_tools"])
        self.assertEqual(agent_cls.return_value.run.call_args.kwargs["tool_names"], ["gen_slides"])
        self.assertEqual(agent_cls.return_value.run.call_args.kwargs["max_tokens"], 1400)

    def test_lead_skips_tools_for_normal_research_answer(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        synthesis = {"overview_md": "syn", "key_points": [], "disagreements": [], "gaps": [], "next_steps": []}
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = "answer"
            app._lead("Summarize the findings", transcript=[], synthesis=synthesis, max_iter=2)
        self.assertFalse(agent_cls.return_value.run.call_args.kwargs["use_tools"])
        self.assertIsNone(agent_cls.return_value.run.call_args.kwargs["tool_names"])

    def test_lead_uses_long_report_prompt_and_more_tokens(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        synthesis = {"overview_md": "syn", "key_points": [], "disagreements": [], "gaps": [], "next_steps": []}
        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
            agent_cls.return_value.run.return_value.text = "answer"
            app._lead("Write a benchmark report", transcript=[], synthesis=synthesis, max_iter=2, long_report_mode=True)
        self.assertEqual(agent_cls.return_value.run.call_args.kwargs["instructions"], LONG_REPORT_LEAD_PROMPT)
        self.assertEqual(agent_cls.return_value.run.call_args.kwargs["max_tokens"], _LONG_REPORT_LEAD_MAX_TOKENS)

    def test_run_turn_structured_appends_reference_links_in_long_report_mode(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        fake_reports = [
            WorkerReport(
                task="task a",
                text="worker a",
                citations=[],
                parsed={
                    "summary": "summary a",
                    "claims": [
                        {
                            "text": "claim a",
                            "citations": [
                                "https://EXAMPLE.com/report?utm_source=news#top",
                                "https://example.com/report",
                            ],
                            "confidence": "high",
                        }
                    ],
                    "gaps": [],
                },
                agent_id="sub-1",
            ),
            WorkerReport(
                task="task b",
                text="worker b",
                citations=[],
                parsed={
                    "summary": "summary b",
                    "claims": [
                        {
                            "text": "claim b",
                            "citations": ["https://example.com/report?utm_medium=social"],
                            "confidence": "medium",
                        }
                    ],
                    "gaps": [],
                },
                agent_id="sub-2",
            ),
        ]
        fake_synthesis = {
            "overview_md": "draft overview",
            "key_points": [
                {
                    "text": "lead point",
                    "citations": ["https://example.com/report?utm_campaign=launch"],
                    "confidence": "low",
                }
            ],
            "disagreements": [],
            "gaps": [],
            "next_steps": [],
        }
        with patch.object(app, "_plan", return_value=["task a", "task b"]):
            with patch.object(app, "_run_workers", return_value=fake_reports):
                with patch.object(app, "_synthesize", return_value=fake_synthesis):
                    with patch.object(app, "_lead", return_value="## Conclusion\n\nLong Report Text"):
                        envelope, _ = app._run_turn_structured(
                            "topic",
                            transcript=[],
                            num_agent=2,
                            max_iter_per_agent=2,
                            long_report_mode=True,
                        )

        self.assertIn("## Reference", envelope["lead"]["overview_md"])
        self.assertEqual(envelope["lead"]["overview_md"].count("## Reference"), 1)
        self.assertIn("- [1] https://example.com/report", envelope["lead"]["overview_md"])
        self.assertNotIn("utm_", envelope["lead"]["overview_md"])

    def test_run_turn_structured_rebuilds_existing_reference_section_in_long_report_mode(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        fake_reports = [
            WorkerReport(
                task="task",
                text="worker",
                citations=[],
                parsed={
                    "summary": "summary",
                    "claims": [
                        {
                            "text": "claim",
                            "citations": ["https://example.com/report?utm_source=news"],
                            "confidence": "high",
                        }
                    ],
                    "gaps": [],
                },
                agent_id="sub-1",
            )
        ]
        fake_synthesis = {"overview_md": "draft", "key_points": [], "disagreements": [], "gaps": [], "next_steps": []}
        lead_text = "## Conclusion\n\nText\n\n## Reference\n\n- https://example.com/already"
        with patch.object(app, "_plan", return_value=["task"]):
            with patch.object(app, "_run_workers", return_value=fake_reports):
                with patch.object(app, "_synthesize", return_value=fake_synthesis):
                    with patch.object(app, "_lead", return_value=lead_text):
                        envelope, _ = app._run_turn_structured(
                            "topic",
                            transcript=[],
                            num_agent=1,
                            max_iter_per_agent=2,
                            long_report_mode=True,
                        )

        self.assertEqual(
            envelope["lead"]["overview_md"],
            "## Conclusion\n\nText\n\n## Reference\n\n- [1] https://example.com/report",
        )

    def test_run_turn_structured_replaces_partial_reference_heading_in_long_report_mode(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        fake_reports = [
            WorkerReport(
                task="task",
                text="worker",
                citations=[],
                parsed={
                    "summary": "summary",
                    "claims": [
                        {
                            "text": "claim",
                            "citations": ["https://example.com/report?utm_source=news"],
                            "confidence": "high",
                        }
                    ],
                    "gaps": [],
                },
                agent_id="sub-1",
            )
        ]
        fake_synthesis = {"overview_md": "draft", "key_points": [], "disagreements": [], "gaps": [], "next_steps": []}
        lead_text = "## Conclusion\n\nText\n\n## Ref"
        with patch.object(app, "_plan", return_value=["task"]):
            with patch.object(app, "_run_workers", return_value=fake_reports):
                with patch.object(app, "_synthesize", return_value=fake_synthesis):
                    with patch.object(app, "_lead", return_value=lead_text):
                        envelope, _ = app._run_turn_structured(
                            "topic",
                            transcript=[],
                            num_agent=1,
                            max_iter_per_agent=2,
                            long_report_mode=True,
                        )

        self.assertEqual(
            envelope["lead"]["overview_md"],
            "## Conclusion\n\nText\n\n## Reference\n\n- [1] https://example.com/report",
        )

    def test_run_turn_structured_uses_raw_report_citations_for_long_report_references(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        fake_reports = [
            WorkerReport(
                task="task",
                text="worker markdown that was not valid json",
                citations=["https://example.com/source?utm_source=news"],
                parsed={
                    "summary": "worker markdown that was not valid json",
                    "claims": [],
                    "gaps": ["non_json_output"],
                },
                agent_id="sub-1",
            )
        ]
        fake_synthesis = {
            "overview_md": "draft overview",
            "key_points": [],
            "disagreements": [],
            "gaps": ["non_json_output"],
            "next_steps": [],
        }
        with patch.object(app, "_plan", return_value=["task"]):
            with patch.object(app, "_run_workers", return_value=fake_reports):
                with patch.object(app, "_synthesize", return_value=fake_synthesis):
                    with patch.object(app, "_lead", return_value="## Conclusion\n\nLead text"):
                        envelope, _ = app._run_turn_structured(
                            "topic",
                            transcript=[],
                            num_agent=1,
                            max_iter_per_agent=2,
                            long_report_mode=True,
                        )

        self.assertIn("## Reference", envelope["lead"]["overview_md"])
        self.assertIn("- [1] https://example.com/source", envelope["lead"]["overview_md"])
        self.assertEqual(envelope["citations_dedup"][0]["canonical_url"], "https://example.com/source")

    def test_chat_session_keeps_full_successful_transcript(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        session = app.session(num_agent=1, max_iter_per_agent=2)
        with patch.object(
            app,
            "_run_turn_structured",
            side_effect=[
                ({"lead": {"overview_md": "first answer"}}, []),
                ({"lead": {"overview_md": "second answer"}}, []),
            ],
        ) as run_turn:
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

    def test_chat_session_ask_detailed_returns_envelope_and_updates_transcript(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        session = app.session(num_agent=1, max_iter_per_agent=2)
        envelope = {
            "version": "research.v1",
            "lead": {"overview_md": "overview"},
            "agents": [],
            "citations": [],
            "citations_dedup": [],
            "meta": {},
        }
        with patch.object(app, "_run_turn_structured", return_value=(envelope, [])) as run_turn:
            result = session.ask_detailed("question")

        self.assertIs(result, envelope)
        self.assertEqual(
            session.transcript,
            [
                ChatMessage(role="user", content="question"),
                ChatMessage(role="assistant", content="overview"),
            ],
        )
        run_turn.assert_called_once()

    def test_canonicalize_url_normalizes_host_and_tracking(self) -> None:
        self.assertEqual(
            _canonicalize_url("HTTPS://Example.com/report?utm_source=news&utm_medium=social#frag"),
            "https://example.com/report",
        )

    def test_should_shortcut_format_follow_up_requires_prior_assistant_answer(self) -> None:
        self.assertFalse(_should_shortcut_format_follow_up("Generate Table", []))
        self.assertFalse(
            _should_shortcut_format_follow_up(
                "Generate Table",
                [ChatMessage(role="user", content="first question")],
            )
        )

    def test_should_shortcut_format_follow_up_rejects_new_research_queries(self) -> None:
        transcript = [
            ChatMessage(role="user", content="Summarize this video"),
            ChatMessage(role="assistant", content="Here is the summary."),
        ]
        self.assertFalse(_should_shortcut_format_follow_up("Search Latest Progress and Generate Table", transcript))
        self.assertFalse(_should_shortcut_format_follow_up("Generate a table for https://example.com", transcript))

    def test_run_turn_structured_shortcuts_format_follow_up_without_research(self) -> None:
        settings = Settings(api_key="x")
        app = DeepFind(settings=settings)
        transcript = [
            ChatMessage(role="user", content="Summarize this Bilibili interview"),
            ChatMessage(role="assistant", content="Summary line 1\n\nSummary line 2"),
        ]
        with patch.object(app, "_plan") as plan:
            with patch.object(app, "_run_workers") as run_workers:
                with patch.object(app, "_synthesize") as synthesize:
                    with patch.object(app, "_lead") as lead:
                        with patch("deepfind.orchestrator.ResponseAgent") as agent_cls:
                            agent_cls.return_value.run.return_value.text = "| Topic | Detail |\n| --- | --- |\n| A | B |"
                            envelope, reports = app._run_turn_structured(
                                "Generate Table",
                                transcript=transcript,
                                num_agent=1,
                                max_iter_per_agent=2,
                            )

        self.assertEqual(reports, [])
        self.assertEqual(envelope["lead"]["overview_md"], "| Topic | Detail |\n| --- | --- |\n| A | B |")
        self.assertEqual(envelope["agents"], [])
        self.assertEqual(envelope["meta"]["shortcut"], "format_follow_up")
        plan.assert_not_called()
        run_workers.assert_not_called()
        synthesize.assert_not_called()
        lead.assert_not_called()
        self.assertEqual(agent_cls.return_value.run.call_args.kwargs["name"], "lead-format")
        self.assertEqual(agent_cls.return_value.run.call_args.kwargs["instructions"], FORMAT_FOLLOWUP_PROMPT)
        self.assertFalse(agent_cls.return_value.run.call_args.kwargs["use_tools"])
