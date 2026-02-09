from typing import TypeAlias
import polars as pl

REAL: TypeAlias = pl.Float32
INTEGER: TypeAlias = pl.Int32
STRING: TypeAlias = pl.Utf8