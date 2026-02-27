from pathlib import Path
from .base import Renderer, ReportBase
from .types import Document
import base64, json
from dataclasses import asdict


class HTMLReportRenderer(Renderer):
    def __init__(
        self,
        report: ReportBase,
        temp_dir: str | Path = "build",
    ):
        super().__init__(report=report, temp_dir=temp_dir)
    
    def copy_reference_files(self, dst_dir: Path) -> None:
        pass

    def render(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.copy_reference_files(dst_dir=path)

        with open("templates/report_template.html", "r", encoding="utf-8") as f:
            template_html = f.read()
        report_dict = asdict(self.report)
        report_json = json.dumps(report_dict, ensure_ascii=False, separators=(",", ":"))
        report_b64 = base64.b64encode(report_json.encode("utf-8")).decode("ascii")
        final_html = template_html.replace("__DOC_PLACEHOLDER__", report_b64)

        with open(path / "index.html", "w", encoding="utf-8") as f:
            f.write(final_html)