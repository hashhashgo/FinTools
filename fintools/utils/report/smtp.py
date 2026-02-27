import smtplib
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
import os


def send_email_with_attachment(
    to_emails: list[str],
    subject: str,
    body: str,
    attachment_path: Path,
    sender_name: str,
    smtp_server: str = os.getenv("SMTP_SERVER", ""),
    smtp_port: int = int(os.getenv("SMTP_PORT", 0)),
    smtp_user: str = os.getenv("SMTP_USER", ""),
    smtp_password: str = os.getenv("SMTP_PASSWORD", ""),
) -> None:
    """发送带附件的电子邮件。

    参数:
        to_emails (list[str]): 收件人电子邮件地址列表。
        subject (str): 邮件主题。
        body (str): 邮件正文内容。
        attachment_path (Path): 附件文件路径。
        sender_name (str): 发件人名称。
        smtp_server (str): SMTP 服务器地址。
        smtp_port (int): SMTP 服务器端口。
        smtp_user (str): SMTP 用户名（发件人邮箱）。
        smtp_password (str): SMTP 密码（发件人邮箱密码）。
    """
    if not all([smtp_server, smtp_port, smtp_user, smtp_password]):
        raise ValueError("SMTP 配置不完整，请检查环境变量。")

    msg = EmailMessage()
    msg["From"] = formataddr((sender_name, smtp_user))
    msg["To"] = ", ".join(to_emails)
    msg["Subject"] = subject
    msg.set_content(body, charset="utf-8")

    with open(attachment_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype=attachment_path.suffix.lstrip("."),
            filename=attachment_path.name
        )

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
