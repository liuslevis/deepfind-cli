import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepfind import send_kindle as send_kindle_module
from deepfind.send_kindle import (
    KindleSendError,
    MissingKindleConfigError,
    attachment_filename,
    sanitize_filename_stem,
    send_to_kindle,
)


def _send(paths, **overrides):
    kwargs = {
        "host": "smtp.qq.com",
        "port": 465,
        "username": "me@qq.com",
        "password": "code",
        "recipient": "me@free.kindle.com",
    }
    kwargs.update(overrides)
    return send_to_kindle(paths, **kwargs)


class SendKindleTests(unittest.TestCase):
    def test_sends_attachment_over_ssl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = Path(tmpdir) / "report.pdf"
            pdf.write_bytes(b"%PDF-1.4 fake")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                smtp_ssl.return_value.__enter__.return_value = client
                data = _send([str(pdf)])

        smtp_ssl.assert_called_once()
        self.assertEqual(smtp_ssl.call_args.args[:2], ("smtp.qq.com", 465))
        client.login.assert_called_once_with("me@qq.com", "code")
        message = client.send_message.call_args.args[0]
        self.assertEqual(message["To"], "me@free.kindle.com")
        self.assertEqual(message["From"], "me@qq.com")
        self.assertEqual(message["Subject"], "report")
        self.assertEqual(
            [part.get_filename() for part in message.iter_attachments()], ["report.pdf"]
        )
        self.assertEqual(data["attachments"], [str(pdf)])

    def test_starttls_when_ssl_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Path(tmpdir) / "note.txt"
            doc.write_text("hi", encoding="utf-8")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP") as smtp:
                smtp.return_value.__enter__.return_value = client
                _send([str(doc)], port=587, use_ssl=False, subject="convert")

        client.starttls.assert_called_once()
        self.assertEqual(client.send_message.call_args.args[0]["Subject"], "convert")

    def test_markdown_converted_to_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Path(tmpdir) / "notes.md"
            doc.write_text("# 标题\n\n正文 **bold**\n", encoding="utf-8")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                smtp_ssl.return_value.__enter__.return_value = client
                data = _send([str(doc)])

        message = client.send_message.call_args.args[0]
        attachments = list(message.iter_attachments())
        self.assertEqual([part.get_filename() for part in attachments], ["notes.pdf"])
        self.assertTrue(attachments[0].get_payload(decode=True).startswith(b"%PDF"))
        self.assertEqual(data["sources"], [str(doc)])
        self.assertEqual(data["converted"], [{"source": str(doc), "pdf": "notes.pdf"}])

    def test_text_converted_to_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Path(tmpdir) / "note.txt"
            doc.write_text("hello kindle", encoding="utf-8")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                smtp_ssl.return_value.__enter__.return_value = client
                _send([str(doc)])

        message = client.send_message.call_args.args[0]
        self.assertEqual(
            [part.get_filename() for part in message.iter_attachments()], ["note.pdf"]
        )

    def test_conversion_workdir_cleaned_up(self) -> None:
        captured: list[Path] = []
        real_convert = send_kindle_module.convert_for_kindle

        def spy(path: Path, output_dir: Path, *, title: str | None = None) -> Path:
            captured.append(Path(output_dir))
            return real_convert(path, output_dir, title=title)

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Path(tmpdir) / "note.md"
            doc.write_text("text", encoding="utf-8")
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL"):
                with patch("deepfind.send_kindle.convert_for_kindle", side_effect=spy):
                    _send([str(doc)])

        self.assertTrue(captured)
        self.assertFalse(captured[0].exists())

    def test_subject_used_as_pdf_metadata_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Path(tmpdir) / "note.md"
            doc.write_text("# hello", encoding="utf-8")
            captured: dict[str, str | None] = {}

            def fake_convert(source: Path, output_path: Path, *, title: str | None = None) -> Path:
                captured["title"] = title
                output_path.write_bytes(b"%PDF-1.4 fake")
                return output_path

            client = MagicMock()
            with patch("deepfind.send_kindle.convert_to_pdf", side_effect=fake_convert):
                with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                    smtp_ssl.return_value.__enter__.return_value = client
                    _send([str(doc)], subject="My Kindle Title")

        self.assertEqual(captured["title"], "My Kindle Title")
        self.assertEqual(client.send_message.call_args.args[0]["Subject"], "My Kindle Title")

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(MissingKindleConfigError) as ctx:
            _send(["a.pdf"], recipient="")
        self.assertIn("KINDLE_EMAIL", str(ctx.exception))

    def test_unsupported_extension_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            blob = Path(tmpdir) / "archive.zip"
            blob.write_bytes(b"x")
            with self.assertRaises(KindleSendError):
                _send([str(blob)])

    def test_missing_file_rejected(self) -> None:
        with self.assertRaises(KindleSendError):
            _send(["/definitely/missing/file.pdf"])

    def test_oversized_attachment_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            big = Path(tmpdir) / "big.pdf"
            big.write_bytes(b"x" * 16)
            with patch("deepfind.send_kindle.MAX_ATTACHMENT_BYTES", 8):
                with self.assertRaises(KindleSendError) as ctx:
                    _send([str(big)])
        self.assertIn("limit", str(ctx.exception))

    def test_smtp_failure_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = Path(tmpdir) / "a.pdf"
            pdf.write_bytes(b"x")
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL", side_effect=OSError("down")):
                with self.assertRaises(KindleSendError) as ctx:
                    _send([str(pdf)])
        self.assertIn("SMTP send failed", str(ctx.exception))


class SanitizeFilenameTests(unittest.TestCase):
    def test_strips_illegal_characters(self) -> None:
        self.assertEqual(
            sanitize_filename_stem('a<b>c:d"e/f\\g|h?i*j'),
            "abcdefghij",
        )

    def test_strips_control_characters_and_collapses_whitespace(self) -> None:
        self.assertEqual(sanitize_filename_stem("Hello\t\n  World\x00"), "Hello World")

    def test_full_width_separators_folded_and_removed(self) -> None:
        self.assertEqual(sanitize_filename_stem("标题：季度／报告"), "标题季度报告")

    def test_keeps_cjk_and_spaces(self) -> None:
        self.assertEqual(sanitize_filename_stem("  2025 年度报告  "), "2025 年度报告")

    def test_trailing_dots_and_spaces_trimmed(self) -> None:
        self.assertEqual(sanitize_filename_stem("report... "), "report")

    def test_all_illegal_title_returns_empty(self) -> None:
        self.assertEqual(sanitize_filename_stem('<>:"/\\|?*'), "")
        self.assertEqual(sanitize_filename_stem(""), "")
        self.assertEqual(sanitize_filename_stem(None), "")

    def test_windows_reserved_names_escaped(self) -> None:
        self.assertEqual(sanitize_filename_stem("CON"), "CON_")
        self.assertEqual(sanitize_filename_stem("com1"), "com1_")

    def test_long_title_truncated(self) -> None:
        stem = sanitize_filename_stem("x" * 400)
        self.assertEqual(len(stem), send_kindle_module.MAX_FILENAME_STEM_LENGTH)

    def test_attachment_filename_keeps_extension(self) -> None:
        self.assertEqual(
            attachment_filename(Path("a/b/report.docx"), "My Report"), "My Report.docx"
        )
        self.assertEqual(attachment_filename(Path("a/b/report.docx"), ""), "report.docx")
        self.assertEqual(attachment_filename(Path("a/b/REPORT.PDF"), "T"), "T.pdf")


class TitleRenameTests(unittest.TestCase):
    def test_title_renames_existing_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = Path(tmpdir) / "report.pdf"
            pdf.write_bytes(b"%PDF-1.4 original")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                smtp_ssl.return_value.__enter__.return_value = client
                data = _send([str(pdf)], title="My Kindle Title")

            message = client.send_message.call_args.args[0]
            attachments = list(message.iter_attachments())
            self.assertEqual(
                [part.get_filename() for part in attachments],
                ["My Kindle Title.pdf"],
            )
            self.assertTrue(attachments[0].get_payload(decode=True).startswith(b"%PDF"))
            self.assertEqual(data["filenames"], ["My Kindle Title.pdf"])
            self.assertTrue(pdf.exists())
            self.assertEqual(pdf.read_bytes(), b"%PDF-1.4 original")

    def test_title_keeps_original_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Path(tmpdir) / "draft.docx"
            doc.write_bytes(b"PK fake")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                smtp_ssl.return_value.__enter__.return_value = client
                _send([str(doc)], title="Quarterly Notes")

        message = client.send_message.call_args.args[0]
        self.assertEqual(
            [part.get_filename() for part in message.iter_attachments()],
            ["Quarterly Notes.docx"],
        )

    def test_title_used_for_converted_pdf_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Path(tmpdir) / "notes.md"
            doc.write_text("# 标题\n\n正文", encoding="utf-8")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                smtp_ssl.return_value.__enter__.return_value = client
                data = _send([str(doc)], title="季度:报告/总结")

        message = client.send_message.call_args.args[0]
        self.assertEqual(
            [part.get_filename() for part in message.iter_attachments()],
            ["季度报告总结.pdf"],
        )
        self.assertEqual(data["converted"], [{"source": str(doc), "pdf": "季度报告总结.pdf"}])

    def test_multiple_files_share_title_without_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "one.pdf"
            second = Path(tmpdir) / "two.pdf"
            first.write_bytes(b"%PDF one")
            second.write_bytes(b"%PDF two")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                smtp_ssl.return_value.__enter__.return_value = client
                _send([str(first), str(second)], title="Same Title")

        message = client.send_message.call_args.args[0]
        self.assertEqual(
            [part.get_filename() for part in message.iter_attachments()],
            ["Same Title.pdf", "Same Title-1.pdf"],
        )

    def test_title_with_only_illegal_chars_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = Path(tmpdir) / "report.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                smtp_ssl.return_value.__enter__.return_value = client
                _send([str(pdf)], title='*/\\:?*')

        message = client.send_message.call_args.args[0]
        self.assertEqual(
            [part.get_filename() for part in message.iter_attachments()],
            ["kindle-document.pdf"],
        )

    def test_title_becomes_default_property_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "report.md"
            source.write_text("# Report", encoding="utf-8")
            converted = Path(tmpdir) / "converted.pdf"
            converted.write_bytes(b"%PDF-1.4")
            with patch(
                "deepfind.send_kindle.convert_for_kindle",
                return_value=converted,
            ) as converter:
                with patch("deepfind.send_kindle.smtplib.SMTP_SSL"):
                    _send([str(source)], title="Doc Title")
        converter.assert_called_once()
        self.assertEqual(converter.call_args.kwargs["title"], "Doc Title")

    def test_no_title_keeps_original_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = Path(tmpdir) / "report.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            client = MagicMock()
            with patch("deepfind.send_kindle.smtplib.SMTP_SSL") as smtp_ssl:
                smtp_ssl.return_value.__enter__.return_value = client
                data = _send([str(pdf)])
        self.assertEqual(
            [part.get_filename() for part in client.send_message.call_args.args[0].iter_attachments()],
            ["report.pdf"],
        )
        self.assertEqual(data["attachments"], [str(pdf)])
        self.assertEqual(data["converted"], [])


if __name__ == "__main__":
    unittest.main()
