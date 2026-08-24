"""Email tool — send mail through the user's configured SMTP account."""

from __future__ import annotations

import asyncio
import logging
import re
import smtplib
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from typing import Any

from app.assistant_settings import EmailSettings, load_assistant_settings
from app.tools.base import (
    BaseTool,
    ToolDanger,
    ToolResult,
    object_schema,
    string_prop,
)

logger = logging.getLogger(__name__)

ADDRESS_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _split_addresses(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = re.split(r"[,;]", str(value))
    return [item.strip() for item in items if item.strip()]


def _recipient_allowed(address: str, settings: EmailSettings) -> bool:
    if not settings.allowed_recipients:
        return True
    lowered = address.lower()
    for rule in settings.allowed_recipients:
        rule = rule.strip().lower()
        if not rule:
            continue
        if rule.startswith("@") and lowered.endswith(rule):
            return True
        if rule == lowered:
            return True
    return False


def _send_sync(message: EmailMessage, settings: EmailSettings) -> None:
    password = settings.resolved_password()
    if settings.smtp_port == 465:
        server: smtplib.SMTP = smtplib.SMTP_SSL(
            settings.smtp_host, settings.smtp_port, timeout=settings.timeout_seconds
        )
    else:
        server = smtplib.SMTP(
            settings.smtp_host, settings.smtp_port, timeout=settings.timeout_seconds
        )
    try:
        server.ehlo()
        if settings.use_tls and settings.smtp_port != 465:
            server.starttls()
            server.ehlo()
        if settings.username and password:
            server.login(settings.username, password)
        server.send_message(message)
    finally:
        try:
            server.quit()
        except Exception:  # noqa: BLE001
            pass


class SendEmailTool(BaseTool):
    name = "send_email"
    description = (
        "Send an email from the user's configured account. Only works once SMTP "
        "is set up in the control panel; always confirm the recipient and subject "
        "with the user before sending."
    )
    danger = ToolDanger.SENSITIVE
    category = "communication"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "to": string_prop("Recipient address, or several separated by commas."),
                "subject": string_prop("Subject line."),
                "body": string_prop("Plain-text message body."),
                "cc": string_prop("Optional CC addresses."),
                "bcc": string_prop("Optional BCC addresses."),
            },
            required=["to", "subject", "body"],
        )

    def availability(self) -> str | None:
        settings = load_assistant_settings().email
        if not settings.enabled:
            return "Email is not enabled in assistant settings"
        if not settings.smtp_host or not settings.from_address:
            return "SMTP host and from-address are not configured"
        return None

    async def run(self, args: dict[str, Any]) -> ToolResult:
        settings = load_assistant_settings().email
        if not settings.enabled:
            return ToolResult.failure(
                "Email is not enabled. Turn it on under Assistant → Email in the control panel."
            )
        if not settings.smtp_host or not settings.from_address:
            return ToolResult.failure("SMTP host and from-address must be configured first.")

        to = _split_addresses(args.get("to"))
        cc = _split_addresses(args.get("cc"))
        bcc = _split_addresses(args.get("bcc"))
        if not to:
            return ToolResult.failure("At least one recipient is required.")

        for address in to + cc + bcc:
            _, parsed = parseaddr(address)
            if not ADDRESS_RE.match(parsed):
                return ToolResult.failure(f"'{address}' is not a valid email address.")
            if not _recipient_allowed(parsed, settings):
                return ToolResult.failure(
                    f"'{parsed}' is not in the allowed-recipients list for this account."
                )

        subject = str(args.get("subject") or "").strip()
        body = str(args.get("body") or "")
        if not subject:
            return ToolResult.failure("A subject is required.")

        message = EmailMessage()
        message["From"] = (
            formataddr((settings.from_name, settings.from_address))
            if settings.from_name
            else settings.from_address
        )
        message["To"] = ", ".join(to)
        if cc:
            message["Cc"] = ", ".join(cc)
        message["Subject"] = subject
        message.set_content(body)

        try:
            await asyncio.to_thread(_send_sync, message, settings)
        except smtplib.SMTPAuthenticationError:
            return ToolResult.failure(
                "SMTP rejected the credentials. Check the username and app password."
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("send_email failed")
            return ToolResult.failure(f"Could not send the email: {exc}")

        recipients = ", ".join(to + cc + bcc)
        return ToolResult.success(
            f"Email '{subject}' sent to {recipients}.",
            data={"to": to, "cc": cc, "bcc": bcc, "subject": subject},
        )


def email_tools() -> list[BaseTool]:
    return [SendEmailTool()]
