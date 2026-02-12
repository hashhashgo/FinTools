from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, TypeAlias, Tuple, List, cast
import pandas as pd
from pandas.core.indexes.accessors import DatetimeProperties
from pathlib import Path

from .types import *

DSL: TypeAlias = str | Tuple | list['DSL']

def _cell_value_to_inlines(v: Any) -> list[Inline]:
    """把一个 DataFrame 单元格值变成 Inline children。"""
    if v is None:
        s = ""
    elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        s = "" if math.isnan(v) else ("∞" if v > 0 else "-∞")
    elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
        s = str(v)
    else:
        s = str(v)
    return [Text(text=s)]

def _infer_align_for_df(df: pd.DataFrame) -> list[Literal["left", "right", "center", "none"]]:
    aligns: list[Literal["left", "right", "center", "none"]] = []
    for col in df.columns:
        s = df[col]
        # pandas 判断 numeric
        if pd.api.types.is_numeric_dtype(s):
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns

def df_to_table_ast(
    df: pd.DataFrame,
    *,
    include_index: bool = False,
    index_name: str = "",
    max_rows: int | None = None,
    max_cols: int | None = None,
    align: list[Literal["left", "right", "center", "none"]] | None = None,
) -> Table:
    """把 pandas.DataFrame 转成 Table AST（你的 Table/TableRow/TableCell 结构）。"""

    work = df.copy()

    # 1) index 作为第一列（可选）
    if include_index:
        idx = work.index
        idx_col_name = (idx.name if idx.name is not None else index_name) or ""
        work.insert(0, idx_col_name if idx_col_name else "index", idx)

    # 2) 截断列
    if max_cols is not None and work.shape[1] > max_cols:
        work = work.iloc[:, :max_cols]

    # 3) 截断行
    if max_rows is not None and work.shape[0] > max_rows:
        work = work.iloc[:max_rows, :]

    # 4) header
    header_cells = [
        TableCell(children=_cell_value_to_inlines(col))
        for col in work.columns.tolist()
    ]
    header = TableRow(cells=header_cells)

    # 5) body
    body: list[TableRow] = []
    # 用 itertuples 比 iterrows 更快
    for row in work.itertuples(index=False, name=None):
        cells = [TableCell(children=_cell_value_to_inlines(v)) for v in row]
        body.append(TableRow(cells=cells))

    # 6) align
    final_align = align if align is not None else _infer_align_for_df(work)

    return Table(align=final_align, header=header, body=body)

def df_to_records_json_safe(df: pd.DataFrame) -> list[dict]:
    df2 = df.copy()

    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            df2[col] = cast(DatetimeProperties, df2[col].dt).strftime("%Y-%m-%dT%H:%M:%S")

    return df2.to_dict(orient="records")

class ReportBase:
    def __init__(self,
                 title: str = "",
                 author: str = "",
                 date: str = "") -> None:
        self.report = Document(
            meta = {
                "title": title,
                "author": author,
                "date": date,
            }
        )
    
    def create_document(self, dsl: DSL) -> None:
        """从 DSL 创建 Document 内容。"""
        out: list[Block] = []
        self.parse_block(dsl, out, level=2)
        self.report.children = out
        
    @staticmethod
    def parse_block(obj: DSL, out: List[Block], level: int = 1) -> None:

        if isinstance(obj, list):
            for x in obj:
                ReportBase.parse_block(cast(DSL, x), out, level)
            return

        if isinstance(obj, str):
            out.append(Paragraph(children=[Text(text=obj)]))
            return

        elif isinstance(obj, tuple) and len(obj) >= 1 and isinstance(obj[0], str):
            opts = {}
            width = 1.0
            aspect_ratio = 2.0
            if obj[0] in ['img', 'chart', 'flowgraph']:
                if len(obj) >= 3:
                    opts = obj[2]
                else: opts = {}
                if not isinstance(opts, dict):
                    raise ValueError(f'img opts must be dict but got {type(opts)}: {obj!r}')
                if opts.get("width") and isinstance(opts["width"], (int, float)):
                    if not (0 < opts["width"] <= 1):
                        raise ValueError(f'img width must be in (0, 1], got {opts["width"]}: {obj!r}')
                    width = float(opts["width"])
                if opts.get("aspect_ratio") and isinstance(opts["aspect_ratio"], (int, float)):
                    if not (0 < opts["aspect_ratio"] <= 10):
                        raise ValueError(f'img aspect_ratio must be in (0, 10], got {opts["aspect_ratio"]}: {obj!r}')
                    aspect_ratio = float(opts["aspect_ratio"])
                elif opts.get("height") and isinstance(opts["height"], (int, float)):
                    if not (0 < opts["height"] <= 1):
                        raise ValueError(f'img height must be in (0, 1], got {opts["height"]}: {obj!r}')
                    aspect_ratio = width / float(opts["height"])

            # ("img", path, opts)
            if obj[0] == "img":
                if len(obj) not in (2, 3) or len(obj) == 1 or not isinstance(obj[1], (str, Path)):
                    raise ValueError(f'img DSL must be ("img", path[, opts]) but got {obj!r}')
                
                path = str(obj[1])

                out.append(
                    Paragraph(children=[
                        Image(
                            src=path,
                            title=cast(Optional[str], opts.get("title")),
                            alt=[Text(text=str(opts.get("alt", "")))] if opts.get("alt") else [],
                            width=width,
                            aspect_ratio=aspect_ratio,
                        )
                    ], align="center")
                )
                return

            if obj[0] == "flowgraph":
                if len(obj) not in (2, 3) or len(obj) == 1:
                    raise ValueError(f'flowgraph DSL must be ("flowgraph", graph[, opts]) but got {obj!r}')
                graph = obj[1]
                options = {}
                if len(obj) == 3:
                    if not isinstance(obj[2], dict):
                        raise ValueError(f'flowgraph opts must be dict but got {type(obj[2])}: {obj!r}')
                    options = obj[2]

                out.append(
                    Paragraph(children=[
                        FlowGraph(
                            title=cast(Optional[str], options.get("title")),
                            graph=graph,
                            options=options,
                            width=width,
                            aspect_ratio=aspect_ratio,
                        )
                    ], align="center")
                )
                return
            
            if obj[0] == "chart" and len(obj) >= 2:
                if len(obj) not in (2, 3) or not isinstance(obj[1], (dict, ChartSpec)):
                    raise ValueError(f'img DSL must be ("chart", spec[, opts]) but got {obj!r}')
                spec = obj[1]
                # 允许 dict 或 ChartSpec
                if isinstance(spec, dict):
                    data = spec.get("data")
                    if isinstance(data, pd.DataFrame):
                        # DataFrame 转 list[dict]
                        spec["data"] = df_to_records_json_safe(data)
                    # 你上面设计的 ChartSpec 字段叫 chart_type / data / encoding
                    chart_spec = ChartSpec(
                        chart_type=spec.get("type", "line"),
                        data=spec.get("data", []),
                        encoding=spec.get("enc", {}),
                        title=spec.get("title"),
                        options=spec.get("options", {}),
                    )
                elif isinstance(spec, ChartSpec):
                    chart_spec = spec
                else:
                    raise ValueError(f'chart expects dict or ChartSpec, got {type(spec)}')

                out.append(
                    Paragraph(children=[
                        ChartInline(
                            spec=chart_spec,
                            width=width,
                            aspect_ratio=aspect_ratio,
                        )
                    ], align="center")
                )
                return

            if obj[0] == "table" and len(obj) == 2:
                out.append(df_to_table_ast(obj[1]))
                return

            if obj[0] == "list" and len(obj) == 2 and isinstance(obj[1], list):
                items: list[ListItem] = []
                for x in obj[1]:
                    sub = []
                    ReportBase.parse_block(x, sub, level)
                    items.append(ListItem(children=sub))
                out.append(ListBlock(items = items))
                return
            
            # ("Title", [...])
            if len(obj) == 2 and isinstance(obj[0], str) and isinstance(obj[1], list):
                title, children = obj
                out.append(Heading(level=level, children=[Text(text=title)]))
                for c in children:
                    ReportBase.parse_block(c, out, level + 1)
                return

        raise ValueError(f"Unsupported DSL element: {obj}")
    
    def get_document(self) -> Document:
        """获取生成的 Document 对象。"""
        return self.report
    
class Renderer(ABC):

    def __init__(
        self,
        report: ReportBase,
        temp_dir: str | Path = "build",
    ):
        self.report = report.get_document()
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def render(self, path: Path) -> None:
        """渲染报告到指定路径"""
        raise NotImplementedError()