from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, TypeAlias, Optional, Union, Any, List, Tuple


# ======================
# Inline nodes (行内)
# ======================

@dataclass
class Text:
    t: Literal["text"] = "text"
    text: str = ""

@dataclass
class SoftBreak:
    t: Literal["softbreak"] = "softbreak"

@dataclass
class HardBreak:
    t: Literal["hardbreak"] = "hardbreak"

@dataclass
class Emph:
    t: Literal["emph"] = "emph"
    children: list[Inline] = field(default_factory=list)

@dataclass
class Strong:
    t: Literal["strong"] = "strong"
    children: list[Inline] = field(default_factory=list)

@dataclass
class Strikethrough:
    # GFM 扩展
    t: Literal["strike"] = "strike"
    children: list[Inline] = field(default_factory=list)

@dataclass
class CodeSpan:
    t: Literal["codespan"] = "codespan"
    code: str = ""

@dataclass
class Link:
    t: Literal["link"] = "link"
    href: str = ""
    title: str | None = None
    children: list[Inline] = field(default_factory=list)

@dataclass
class Image:
    t: Literal["image"] = "image"
    src: str = ""
    title: str | None = None
    alt: list[Inline] = field(default_factory=list)
    width: float = 1 # * linewidth (max-width)
    aspect_ratio: float = 1  # width / height (max-height)

@dataclass
class FlowGraph:
    t: Literal["flowgraph"] = "flowgraph"
    title: str | None = None
    graph: List[Tuple[Any, ...]] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    width: float = 1. # * linewidth (max-width)
    aspect_ratio: float = 2.  # width / height (max-height)

@dataclass
class ChartSpec:
    """
    JSON-friendly chart description.
    data: list[dict] 是最稳的（可 json.dumps / 可传前端 / 可存 DB）
    encoding: 描述 x/y/series 等映射
    """
    chart_type: Literal["line", "bar", "scatter", "candlestick", "pie"] = "line"
    data: list[dict[str, Any]] = field(default_factory=list)

    # 最小编码：x 列名；y 可以是一个列名或多个列名；series 可选（分组）
    encoding: dict[str, Any] = field(default_factory=dict)

    title: str | None = None
    options: dict[str, Any] = field(default_factory=dict)  # 透传给 renderer 的可选项

@dataclass
class ChartInline:
    # 和 Image 一样是 inline
    t: Literal["chart"] = "chart"
    spec: ChartSpec = field(default_factory=ChartSpec)

    # 类似 image 的 title/alt，方便无图环境降级/可访问性/日志
    title: str | None = None
    alt: list[Inline] = field(default_factory=list)

    # inline 尺寸提示（最终渲染时决定怎么用）
    width: float = 1.0 # * linewidth
    aspect_ratio: float = 2.0  # width / height

@dataclass
class InlineMath:
    # 可选扩展：$...$
    t: Literal["inlinemath"] = "inlinemath"
    latex: str = ""

@dataclass
class FootnoteRef:
    # 可选扩展：[^id]
    t: Literal["footnote_ref"] = "footnote_ref"
    ref: str = ""


Inline: TypeAlias = Union[
    Text,
    SoftBreak,
    HardBreak,
    Emph,
    Strong,
    Strikethrough,
    CodeSpan,
    Link,
    Image,
    InlineMath,
    FootnoteRef,
    ChartInline,
    FlowGraph,
]


# ======================
# Block nodes (块级)
# ======================

@dataclass
class Paragraph:
    t: Literal["paragraph"] = "paragraph"
    children: list[Inline] = field(default_factory=list)
    align: Literal["left", "right", "center", "justify"] = "left"

@dataclass
class Heading:
    t: Literal["heading"] = "heading"
    level: int = 1  # 1..6
    children: list[Inline] = field(default_factory=list)

@dataclass
class BlockQuote:
    t: Literal["blockquote"] = "blockquote"
    children: list[Block] = field(default_factory=list)

@dataclass
class ThematicBreak:
    # --- / *** / ___
    t: Literal["hr"] = "hr"

@dataclass
class CodeBlock:
    t: Literal["codeblock"] = "codeblock"
    code: str = ""
    info: str | None = None  # ```python 里的 python
    meta: dict[str, Any] | None = None

@dataclass
class HtmlBlock:
    # CommonMark 支持 HTML block；你也可以选择禁用它
    t: Literal["htmlblock"] = "htmlblock"
    html: str = ""

@dataclass
class ListItem:
    t: Literal["list_item"] = "list_item"
    children: list[Block] = field(default_factory=list)
    checked: bool | None = None  # GFM task list: - [x] / - [ ]

@dataclass
class ListBlock:
    t: Literal["list"] = "list"
    ordered: bool = False
    start: int | None = None         # ordered list 起始编号
    tight: bool = False              # tight/loose（渲染时可用）
    items: list[ListItem] = field(default_factory=list)

@dataclass
class TableCell:
    t: Literal["table_cell"] = "table_cell"
    children: list[Inline] = field(default_factory=list)

@dataclass
class TableRow:
    t: Literal["table_row"] = "table_row"
    cells: list[TableCell] = field(default_factory=list)

@dataclass
class Table:
    # GFM 表格扩展
    t: Literal["table"] = "table"
    align: list[Literal["left", "right", "center", "none"]] = field(default_factory=list)
    header: TableRow | None = None
    body: list[TableRow] = field(default_factory=list)

@dataclass
class FootnoteDef:
    # 可选扩展：[^id]: ....
    t: Literal["footnote_def"] = "footnote_def"
    ref: str = ""
    children: list[Block] = field(default_factory=list)

@dataclass
class Document:
    t: Literal["doc"] = "doc"
    children: list[Block] = field(default_factory=list)
    meta: dict[str, Any] | None = None  # 可放 title/author/date 等


Block: TypeAlias = Union[
    Paragraph,
    Heading,
    BlockQuote,
    ThematicBreak,
    CodeBlock,
    HtmlBlock,
    ListBlock,
    Table,
    FootnoteDef,
]
