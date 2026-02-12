from typing import Sequence, cast
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
from io import BytesIO

from reportlab.pdfbase import pdfmetrics

from reportlab.lib.colors import black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
import reportlab.platypus as platypus
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.lib.styles import StyleSheet1, ParagraphStyle, ListStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .base import ReportBase, Renderer
from .types import FlowGraph, Heading, Paragraph, Table, TableRow, Image, Text, ChartInline

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import graphviz


def _plot_by_type(ax, chart_type: str, x: Sequence, y: Sequence, label: str | None = None) -> None:
    if chart_type == "line":
        ax.plot(x, y, label=label)
    elif chart_type == "bar":
        ax.bar(x, y, label=label)
    elif chart_type == "scatter":
        ax.scatter(x, y, label=label)
    elif chart_type == "candlestick":
        # 占位：通常需要 open/high/low/close 四列
        raise NotImplementedError("candlestick matplotlib rendering not implemented yet")
    elif chart_type == "pie":
        ax.pie(y, labels=x, autopct='%1.1f%%')
    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

def chart_to_matplotlib_figure(chart: ChartInline, width: float, height: float, dpi: int = 150) -> Figure:
    spec = chart.spec
    df = pd.DataFrame(spec.data)

    x = spec.encoding.get("x")
    y = spec.encoding.get("y")
    series = spec.encoding.get("series")  # 可选：按列分组多条线
    options = spec.options

    if x is None or y is None:
        raise ValueError("ChartSpec.encoding must include 'x' and 'y'")

    ys: list[str] = [y] if isinstance(y, str) else list(y)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    try: df[x] = pd.to_datetime(df[x], format="%Y-%m-%dT%H:%M:%S")
    except Exception: pass

    is_legend = False
    if series:
        # 按 series 分组，每组画 ys
        for key, g in df.groupby(series):
            for col in ys:
                label = f"{key}:{col}" if len(ys) > 1 else str(key)
                _plot_by_type(ax, spec.chart_type, g[x], g[col], label=label)
        is_legend = True
    else:
        for col in ys:
            _plot_by_type(ax, spec.chart_type, df[x], cast(Sequence, df[col]), label=col)
        if len(ys) > 1:
            is_legend = True
    
    if options:
        if "xlabel" in options:
            ax.set_xlabel(options["xlabel"])
        if "ylabel" in options:
            ax.set_ylabel(options["ylabel"])
        if options.get("legend", False) and is_legend:
            ax.legend()

    if spec.title:
        ax.set_title(spec.title)
    
    fig.tight_layout()

    return fig

def chart_to_png_bytes(chart: ChartInline, width: float, height: float, dpi: int = 200) -> BytesIO:
    fig = chart_to_matplotlib_figure(chart, width / inch, height / inch, dpi)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def flowgraph_to_png_bytes(flowgraph: FlowGraph, width: float, height: float, dpi: int = 300) -> BytesIO:
    dot = graphviz.Digraph(comment=flowgraph.title, format='png')
    dot.attr(rankdir=flowgraph.options.get("rankdir", "TB"))
    dot.attr(size=f"{int(width / inch)},{int(height / inch)}")
    dot.attr(dpi=str(dpi))
    for item in flowgraph.graph:
        if len(item) < 2:
            raise ValueError(f'flowgraph item must have at least 2 elements, got {item!r}')
        eval(f"dot.{item[0]}")(*item[1:])
    buf = BytesIO(dot.pipe(format='png'))
    buf.seek(0)
    return buf

class PDFReportRenderer(Renderer):
    def __init__(
        self,
        report: ReportBase,
        temp_dir: str | Path = "build",
        regular_font = TTFont("CJK-Regular", "fonts/NotoSansSC-Regular.ttf"),
        bold_font = TTFont("CJK-Bold", "fonts/NotoSansSC-Bold.ttf"),
    ):
        super().__init__(report, temp_dir)
        self.regular_font = regular_font
        self.bold_font = bold_font
        self.styles = self.get_style_sheet()
        self.temp_files = []
    
    def get_style_sheet(self) -> StyleSheet1:
        pdfmetrics.registerFont(self.regular_font)
        pdfmetrics.registerFont(self.bold_font)
        stylesheet = StyleSheet1()
        stylesheet.add(ParagraphStyle(name='Normal',
                                    fontName=self.regular_font.fontName,
                                    fontSize=10,
                                    leading=12)
                    )

        stylesheet.add(ParagraphStyle(name='BodyText',
                                    parent=stylesheet['Normal'],
                                    spaceBefore=6)
                    )

        stylesheet.add(ParagraphStyle(name='Heading1',
                                    parent=stylesheet['Normal'],
                                    fontName = self.bold_font.fontName,
                                    fontSize=24,
                                    leading=22,
                                    spaceAfter=6),
                    alias='h1')

        stylesheet.add(ParagraphStyle(name='Title',
                                    parent=stylesheet['Normal'],
                                    fontName = self.bold_font.fontName,
                                    fontSize=24,
                                    leading=22,
                                    alignment=TA_CENTER,
                                    spaceAfter=6),
                    alias='title')

        stylesheet.add(ParagraphStyle(name='Heading2',
                                    parent=stylesheet['Normal'],
                                    fontName = self.bold_font.fontName,
                                    fontSize=20,
                                    leading=18,
                                    spaceBefore=12,
                                    spaceAfter=6,),
                    alias='h2')

        stylesheet.add(ParagraphStyle(name='Heading3',
                                    parent=stylesheet['Normal'],
                                    fontName = self.bold_font.fontName,
                                    fontSize=16,
                                    leading=14,
                                    spaceBefore=12,
                                    spaceAfter=6),
                    alias='h3')

        stylesheet.add(ParagraphStyle(name='Heading4',
                                    parent=stylesheet['Normal'],
                                    fontName = self.bold_font.fontName,
                                    fontSize=14,
                                    leading=12,
                                    spaceBefore=10,
                                    spaceAfter=4),
                    alias='h4')

        stylesheet.add(ParagraphStyle(name='Heading5',
                                    parent=stylesheet['Normal'],
                                    fontName = self.bold_font.fontName,
                                    fontSize=12,
                                    leading=10.8,
                                    spaceBefore=8,
                                    spaceAfter=4),
                    alias='h5')

        stylesheet.add(ParagraphStyle(name='Heading6',
                                    parent=stylesheet['Normal'],
                                    fontName = self.bold_font.fontName,
                                    fontSize=12,
                                    leading=8.4,
                                    spaceBefore=6,
                                    spaceAfter=2),
                    alias='h6')

        stylesheet.add(ParagraphStyle(name='Bullet',
                                    parent=stylesheet['Normal'],
                                    firstLineIndent=0,
                                    spaceBefore=3),
                    alias='bu')

        stylesheet.add(ParagraphStyle(name='Definition',
                                    parent=stylesheet['Normal'],
                                    firstLineIndent=0,
                                    leftIndent=36,
                                    bulletIndent=0,
                                    spaceBefore=6,
                                    bulletFontName=self.bold_font.fontName),
                    alias='df')

        stylesheet.add(ParagraphStyle(name='Code',
                                    parent=stylesheet['Normal'],
                                    fontName='Courier',
                                    fontSize=8,
                                    leading=8.8,
                                    firstLineIndent=0,
                                    leftIndent=36,
                                    hyphenationLang=''))

        stylesheet.add(ListStyle(name='UnorderedList',
                                    parent=None,
                                    leftIndent=18,
                                    rightIndent=0,
                                    bulletAlign='left',
                                    bulletType='1',
                                    bulletColor=black,
                                    bulletFontName='Helvetica',
                                    bulletFontSize=12,
                                    bulletOffsetY=0,
                                    bulletDedent='auto',
                                    bulletDir='ltr',
                                    bulletFormat=None,
                                    #start='circle square blackstar sparkle disc diamond'.split(),
                                    start=None,
                                ),
                    alias='ul')

        stylesheet.add(ListStyle(name='OrderedList',
                                    parent=None,
                                    leftIndent=18,
                                    rightIndent=0,
                                    bulletAlign='left',
                                    bulletType='1',
                                    bulletColor=black,
                                    bulletFontName='Helvetica',
                                    bulletFontSize=12,
                                    bulletOffsetY=0,
                                    bulletDedent='auto',
                                    bulletDir='ltr',
                                    bulletFormat=None,
                                    #start='1 a A i I'.split(),
                                    start=None,
                                ),
                    alias='ol')
        return stylesheet

    def render(self, path: Path) -> None:
        os.makedirs(path.parent, exist_ok=True)
        doc = platypus.SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        elements = []
        
        if self.report.meta and  self.report.meta.get("title"):
            elements.append(platypus.Paragraph(self.report.meta["title"], self.styles['Title']))
        
        for block in self.report.children:
            if isinstance(block, Heading):
                if not (block.children and len(block.children) == 1 and isinstance(block.children[0], Text)):
                    raise NotImplementedError("Heading block must have one Text child now.")
                elements.append(platypus.Paragraph(block.children[0].text, self.styles[f'Heading{block.level}']))
            elif isinstance(block, Paragraph):
                for child in block.children:
                    if isinstance(child, Text):
                        elements.append(platypus.Paragraph(
                            child.text,
                            ParagraphStyle(name='BodyText',
                                parent=self.styles['BodyText'],
                                alignment= block.align
                            )
                        ))
                    elif isinstance(child, Image):
                        width = doc.width * child.width
                        height = width / child.aspect_ratio
                        if block.align == "left": align = "LEFT"
                        elif block.align == "center": align = "CENTER"
                        elif block.align == "right": align = "RIGHT"
                        elif block.align == "justify": align = "CENTER"
                        else: raise ValueError(f"Unsupported paragraph align value: {block.align}")
                        elements.append(platypus.Image(child.src, width=width, height=height, kind="bound", hAlign=align))
                    elif isinstance(child, ChartInline):
                        width = child.width * doc.width
                        height = width / child.aspect_ratio
                        elements.append(platypus.Image(
                            chart_to_png_bytes(child, width=width, height=height),
                            width=width,
                            height=height,
                            kind="bound",
                            hAlign="CENTER"
                        ))
                    elif isinstance(child, FlowGraph):
                        width = child.width * doc.width
                        height = width / child.aspect_ratio
                        elements.append(platypus.Image(
                            flowgraph_to_png_bytes(child, width=width, height=height),
                            width=width,
                            height=height,
                            kind="bound",
                            hAlign="CENTER"
                        ))
                    else:
                        raise NotImplementedError(f"Unsupported Paragraph child type: {type(child)}")
            elif isinstance(block, Table):
                data = []
                table_style = [
                    ('FONTNAME', (0, 0), (-1, 0), self.bold_font.fontName),
                    ('FONTNAME', (0, 1), (-1, -1), self.regular_font.fontName),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                ]
                def make_row(row: TableRow) -> list:
                    row_data = []
                    for cell in row.cells:
                        if not (cell.children and len(cell.children) == 1 and isinstance(cell.children[0], Text)):
                            raise NotImplementedError("Table cell must have one Text child now.")
                        row_data.append(cell.children[0].text)
                    return row_data
                cols = 0
                if block.header:
                    data.append(make_row(block.header))
                    table_style.append(('ALIGN', (0, 0), (-1, 0), 'CENTER'))
                    cols = len(block.header.cells)
                for row in block.body:
                    data.append(make_row(row))
                    if cols == 0: cols = len(row.cells)
                    elif cols != len(row.cells):
                        raise ValueError("All table rows must have the same number of cells.")
                if block.align:
                    if len(block.align) != cols:
                        raise ValueError("Table align length must match number of columns.")
                    for i, align in enumerate(block.align):
                        if not align in ('left', 'right', 'center'):
                            raise ValueError(f"Unsupported table align value: {align}")
                        table_style.append(('ALIGN', (i, 0), (i, -1), align.upper()))
                table = platypus.Table(data, repeatRows=1)
                table.setStyle(platypus.TableStyle(table_style))
                elements.append(platypus.Spacer(1, 12))
                elements.append(table)
                elements.append(platypus.Spacer(1, 12))
            else:
                raise NotImplementedError(f"Unsupported block type: {type(block)}")
        doc.build(elements)

