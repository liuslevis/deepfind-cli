from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import deepfind.asr as asr


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def synchronize(self, index: int) -> None:
        return None

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def ipc_collect(self) -> None:
        return None

    def device(self, index: int) -> "_FakeCuda":
        return self

    def __enter__(self) -> "_FakeCuda":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class IdleUnloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._slots = patch.object(asr, "_ASR_SLOTS", [None, None])
        self._slots.start()
        self.addCleanup(self._slots.stop)
        self._reaper = patch.object(asr, "_start_unload_reaper")
        self._reaper.start()
        self.addCleanup(self._reaper.stop)
        fake_torch = _FakeTorch()
        self._modules = patch.dict(sys.modules, {"torch": fake_torch})  # type: ignore[arg-type]
        self._modules.start()
        self.addCleanup(self._modules.stop)
        self.fake_torch = fake_torch

    def _load(self, model_name: str = "") -> tuple[str, object, None, str]:
        return "qwen3_asr", object(), None, "cuda"

    def test_model_is_reused_within_slot(self) -> None:
        with patch.object(asr, "load_model", side_effect=self._load) as load:
            with asr.gpu_asr_slot():
                first = asr._asr_slot_model(asr.slot_index(), "Qwen/Qwen3-ASR-1.7B")
            with asr.gpu_asr_slot():
                second = asr._asr_slot_model(asr.slot_index(), "Qwen/Qwen3-ASR-1.7B")

        self.assertIs(first[1], second[1])
        load.assert_called_once()

    def test_model_switch_releases_previous_model(self) -> None:
        loads = [
            ("qwen3_asr", object(), None, "cuda"),
            ("qwen3_asr", object(), None, "cuda"),
        ]
        with patch.object(asr, "load_model", side_effect=loads) as load:
            with asr.gpu_asr_slot():
                first = asr._asr_slot_model(asr.slot_index(), "Qwen/Qwen3-ASR-1.7B")
            with asr.gpu_asr_slot():
                second = asr._asr_slot_model(asr.slot_index(), "Qwen/Qwen3-ASR-0.6B")

        self.assertIsNot(first[1], second[1])
        self.assertEqual(load.call_count, 2)
        self.assertEqual(self.fake_torch.cuda.empty_cache_calls, 1)
        self.assertEqual(asr._ASR_SLOTS[0].asr_model, "Qwen/Qwen3-ASR-0.6B")

    def test_idle_model_released_after_ttl(self) -> None:
        with patch.object(asr, "load_model", side_effect=self._load):
            with asr.gpu_asr_slot():
                asr._asr_slot_model(asr.slot_index(), "Qwen/Qwen3-ASR-1.7B")

        with patch.object(asr, "_MONOTONIC", return_value=asr._ASR_SLOTS[0].last_used + 3599):
            self.assertEqual(asr.unload_idle_models(), [])
            self.assertIsNotNone(asr._ASR_SLOTS[0])

        with patch.object(asr, "_MONOTONIC", return_value=asr._ASR_SLOTS[0].last_used + 3600):
            self.assertEqual(asr.unload_idle_models(), ["Qwen/Qwen3-ASR-1.7B"])

        self.assertIsNone(asr._ASR_SLOTS[0])
        self.assertGreaterEqual(self.fake_torch.cuda.empty_cache_calls, 1)
        self.assertEqual(asr._ASR_UNLOAD_EVENTS[-1][1:], ("Qwen/Qwen3-ASR-1.7B", True))

    def test_ttl_env_override_and_disable(self) -> None:
        with patch.dict("os.environ", {"ASR_IDLE_UNLOAD_SECONDS": "1800"}, clear=False):
            self.assertEqual(asr.asr_model_idle_unload_seconds(), 1800)
        with patch.dict("os.environ", {"ASR_IDLE_UNLOAD_SECONDS": "0"}, clear=False):
            self.assertIsNone(asr.asr_model_idle_unload_seconds())
        with patch.dict("os.environ", {"ASR_IDLE_UNLOAD_SECONDS": "bogus"}, clear=False):
            self.assertEqual(
                asr.asr_model_idle_unload_seconds(),
                asr.DEFAULT_ASR_IDLE_UNLOAD_SECONDS,
            )

    def test_disabled_ttl_never_unloads(self) -> None:
        with patch.object(asr, "load_model", side_effect=self._load):
            with asr.gpu_asr_slot():
                asr._asr_slot_model(asr.slot_index(), "Qwen/Qwen3-ASR-1.7B")

        with patch.dict("os.environ", {"ASR_IDLE_UNLOAD_SECONDS": "0"}, clear=False):
            with patch.object(asr, "_MONOTONIC", return_value=asr._ASR_SLOTS[0].last_used + 1e6):
                self.assertEqual(asr.unload_idle_models(), [])
        self.assertIsNotNone(asr._ASR_SLOTS[0])

    def test_release_from_reaper_waits_for_busy_slot(self) -> None:
        import threading

        started = threading.Event()
        finished = threading.Event()

        def release_from_other_thread() -> None:
            with asr.gpu_asr_slot():
                started.set()
                finished.wait(5)

        worker = threading.Thread(target=release_from_other_thread)
        worker.start()
        try:
            self.assertTrue(started.wait(5))
            # The reaper must block until the in-flight transcription releases the slot.
            asr.release_gpu_memory()
            self.assertEqual(self.fake_torch.cuda.empty_cache_calls, 1)
        finally:
            finished.set()
            worker.join(5)

    def test_transcribe_segments_reuses_cached_model(self) -> None:
        calls: list[str] = []
        model = object()

        def fake_transcribe(segment, backend, model_arg, processor, device):
            calls.append(segment.name)
            return "hello"

        segments = [
            type("P", (), {"name": "seg_000.wav"})(),
            type("P", (), {"name": "seg_001.wav"})(),
        ]
        with patch.object(asr, "load_model", side_effect=self._load) as load:
            with patch.object(asr, "transcribe_audio", side_effect=lambda *a, **k: fake_transcribe(*a)):
                self.assertEqual(asr.transcribe_segments(segments), "hello\nhello")

        load.assert_called_once()
        self.assertEqual(calls, ["seg_000.wav", "seg_001.wav"])


if __name__ == "__main__":
    unittest.main()
