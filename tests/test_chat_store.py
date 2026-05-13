from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deepfind.chat_store import ChatStore
from deepfind.web_models import WebMessage


class ChatStoreTests(unittest.TestCase):
    def test_create_save_load_and_delete_chat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChatStore(Path(temp_dir))
            chat = store.create_chat("Workspace")
            message = WebMessage(
                id="msg_1",
                role="user",
                content="hello world",
                created_at="2026-03-22T00:00:00+00:00",
                mode="fast",
            )
            updated = chat.model_copy(deep=True)
            updated.messages.append(message)
            updated.updated_at = message.created_at
            store.save_chat(updated)

            reloaded = ChatStore(Path(temp_dir)).get_chat(chat.id)
            self.assertEqual(reloaded.title, "Workspace")
            self.assertEqual(reloaded.messages[0].content, "hello world")

            summaries = store.list_chats()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].preview, "hello world")

            store.delete_chat(chat.id)
            self.assertEqual(store.list_chats(), [])

    def test_get_chat_normalizes_legacy_cloud_model_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = {
                "id": "chat_legacy",
                "title": "Legacy",
                "created_at": "2026-03-22T00:00:00+00:00",
                "updated_at": "2026-03-22T00:00:00+00:00",
                "messages": [
                    {
                        "id": "msg_1",
                        "role": "assistant",
                        "content": "legacy",
                        "created_at": "2026-03-22T00:00:00+00:00",
                        "mode": "fast",
                        "sources": [],
                        "artifacts": [],
                        "model_target": "cloud",
                        "model_label": "qwen3-max",
                    }
                ],
            }
            (root / "chat_legacy.json").write_text(json.dumps(payload), encoding="utf-8")

            chat = ChatStore(root).get_chat("chat_legacy")

        self.assertEqual(chat.messages[0].model_target, "qwen")
