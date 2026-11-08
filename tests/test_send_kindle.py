import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepfind.send_kindle import (
    KindleSendError,
    MissingKindleConfigError,
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


if __name__ == "__main__":
    unittest.main()
