from __future__ import annotations

import mimetypes
import shutil
import smtplib
import ssl
import tempfile
from email.message import EmailMessage
from pathlib import Path
from typing import Sequence

from .kindle_pdf import (
    CONVERTIBLE_EXTENSIONS,
    PdfConversionError,
    convert_to_pdf,
)


DEFAULT_SMTP_HOST = "smtp.qq.com"
DEFAULT_SMTP_PORT = 465
DEFAULT_SMTP_TIMEOUT = 60
MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024

# Formats accepted by the Send to Kindle service.
SUPPORTED_EXTENSIONS = {
    ".azw",
    ".doc",
    ".docx",
    ".epub",
    ".htm",
    ".html",
    ".jpeg",
    ".jpg",
    ".markdown",
    ".md",
    ".mobi",
    ".pdf",
    ".png",
    ".rtf",
    ".txt",
}

REPO_ROOT = Path(__file__).resolve().parent.parent


class KindleSendError(RuntimeError):
    """Base error for Send to Kindle failures."""


class MissingKindleConfigError(KindleSendError):
    """Raised when SMTP or Kindle settings are incomplete."""


def resolve_attachment(file_path: str) -> Path:
    raw = (file_path or "").strip()
    if not raw:
        raise KindleSendError("file_path cannot be empty.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.is_file():
        raise KindleSendError(f"Attachment not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise KindleSendError(
            f"Unsupported attachment type '{path.suffix or path.name}'. "
            f"Kindle accepts: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    if path.stat().st_size > MAX_ATTACHMENT_BYTES:
        raise KindleSendError(
            f"Attachment {path.name} exceeds the 50 MB Send to Kindle limit."
        )
    return path


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def convert_for_kindle(path: Path, output_dir: Path) -> Path:
    """Convert txt/markdown sources to a Kindle-sized PDF; pass others through."""

    if path.suffix.lower() not in CONVERTIBLE_EXTENSIONS:
        return path
    target = _unique_path(output_dir, path.stem, ".pdf")
    try:
        return convert_to_pdf(path, target)
    except PdfConversionError as exc:
        raise KindleSendError(str(exc)) from exc


def _attach(message: EmailMessage, path: Path) -> None:
    mime_type, _ = mimetypes.guess_type(path.name)
    maintype, _, subtype = (mime_type or "application/octet-stream").partition("/")
    message.add_attachment(
        path.read_bytes(),
        maintype=maintype,
        subtype=subtype or "octet-stream",
        filename=path.name,
    )


def send_to_kindle(
    file_paths: Sequence[str],
    *,
    host: str | None,
    port: int,
    username: str | None,
    password: str | None,
    recipient: str | None,
    sender: str | None = None,
    use_ssl: bool = True,
    subject: str | None = None,
    body: str | None = None,
    timeout: int = DEFAULT_SMTP_TIMEOUT,
) -> dict[str, object]:
    smtp_host = (host or "").strip() or DEFAULT_SMTP_HOST
    smtp_user = (username or "").strip()
    smtp_password = (password or "").strip()
    to_address = (recipient or "").strip()
    from_address = (sender or "").strip() or smtp_user

    missing = [
        name
        for name, value in (
            ("SMTP_USER", smtp_user),
            ("SMTP_PASSWORD", smtp_password),
            ("KINDLE_EMAIL", to_address),
        )
        if not value
    ]
    if missing:
        raise MissingKindleConfigError(
            f"Set {', '.join(missing)} in .env to use send_kindle."
        )

    if not file_paths:
        raise KindleSendError("Provide at least one file to send.")
    sources = [resolve_attachment(item) for item in file_paths]

    workdir: Path | None = None
    if any(path.suffix.lower() in CONVERTIBLE_EXTENSIONS for path in sources):
        workdir = Path(tempfile.mkdtemp(prefix="deepfind-kindle-"))
    try:
        attachments = [
            convert_for_kindle(path, workdir) if workdir else path for path in sources
        ]
        return _deliver(
            attachments=attachments,
            sources=sources,
            smtp_host=smtp_host,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            from_address=from_address,
            to_address=to_address,
            port=port,
            use_ssl=use_ssl,
            subject=subject,
            body=body,
            timeout=timeout,
        )
    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)


def _deliver(
    *,
    attachments: Sequence[Path],
    sources: Sequence[Path],
    smtp_host: str,
    smtp_user: str,
    smtp_password: str,
    from_address: str,
    to_address: str,
    port: int,
    use_ssl: bool,
    subject: str | None,
    body: str | None,
    timeout: int,
) -> dict[str, object]:
    total_bytes = sum(path.stat().st_size for path in attachments)
    if total_bytes > MAX_ATTACHMENT_BYTES:
        raise KindleSendError(
            "Total attachment size exceeds the 50 MB Send to Kindle limit."
        )

    message = EmailMessage()
    message["From"] = from_address
    message["To"] = to_address
    message["Subject"] = (subject or "").strip() or attachments[0].stem
    message.set_content((body or "").strip() or "Sent by DeepFind.")
    for path in attachments:
        _attach(message, path)

    smtp_port = int(port or DEFAULT_SMTP_PORT)
    safe_timeout = max(int(timeout or DEFAULT_SMTP_TIMEOUT), 1)
    try:
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(
                smtp_host, smtp_port, timeout=safe_timeout, context=context
            ) as client:
                client.login(smtp_user, smtp_password)
                client.send_message(message)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=safe_timeout) as client:
                client.ehlo()
                client.starttls(context=ssl.create_default_context())
                client.ehlo()
                client.login(smtp_user, smtp_password)
                client.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        raise KindleSendError(
            "SMTP authentication failed. Use the mail provider authorization code as SMTP_PASSWORD."
        ) from exc
    except (smtplib.SMTPException, OSError) as exc:
        raise KindleSendError(f"SMTP send failed: {exc}") from exc

    return {
        "from": from_address,
        "to": to_address,
        "subject": message["Subject"],
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "use_ssl": use_ssl,
        "attachments": [str(path) for path in attachments],
        "sources": [str(path) for path in sources],
        "converted": [
            {"source": str(source), "pdf": attachment.name}
            for source, attachment in zip(sources, attachments)
            if source != attachment
        ],
        "total_bytes": total_bytes,
    }


__all__ = [
    "DEFAULT_SMTP_HOST",
    "DEFAULT_SMTP_PORT",
    "DEFAULT_SMTP_TIMEOUT",
    "KindleSendError",
    "MAX_ATTACHMENT_BYTES",
    "MissingKindleConfigError",
    "SUPPORTED_EXTENSIONS",
    "convert_for_kindle",
    "resolve_attachment",
    "send_to_kindle",
]
