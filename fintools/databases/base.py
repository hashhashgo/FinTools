import sqlite3
from hashlib import sha1
from contextlib import contextmanager
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, date, timedelta
from pyparsing import ABC, abstractmethod
import pandas as pd

from contextvars import ContextVar

from .utils import _python_value_to_sqlite_value, _sqlite_value_to_pandas_value


Fields = Dict[str, Any]

class BaseDB(ABC):

    table_basename: str
    db_path: str
    connection: ContextVar[Optional[sqlite3.Connection]] = ContextVar(
        "connection", default=None
    )
    tables: Dict[str, Any] = {}

    def get_table_name_and_cursor(self, common_fields: Fields = {}) -> Tuple[str, sqlite3.Cursor]:
        table_name = self._get_table_name(common_fields=common_fields)
        
        if not self.tables.get(table_name): self.tables[table_name] = self._get_table_info(common_fields=common_fields)
        if not self.tables.get(table_name):
            raise ValueError(f"表 {table_name} 不存在，请先创建。")
        
        cur = self._get_cursor()
        return table_name, cur
    
    def format_dataframe(self, data: List[Any] | pd.DataFrame, common_fields: Fields = {}) -> pd.DataFrame:
        """
        将数据格式化为 DataFrame。

        参数：
            data: 待格式化的数据，可以是列表或 DataFrame。
            common_fields: 公共字段，如 freq 等，common_fields 作为表名的一部分
        返回值：
            格式化后的 DataFrame。
        """
        table_name = self._get_table_name(common_fields=common_fields)
        
        if not self.tables.get(table_name): self.tables[table_name] = self._get_table_info(common_fields=common_fields)
        if not self.tables.get(table_name):
            raise ValueError(f"表 {table_name} 不存在，请先创建。")
        
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data, columns=data[0].keys() if data else [])

        if not df.empty:
            return _sqlite_value_to_pandas_value(df, type_dict=self.tables[table_name])
        else:
            return df

    def list_all_cached(self, common_fields: Fields = {}) -> List[Any]:
        """
        列出所有缓存的数据条目。

        参数：
            common_fields: 公共字段，如 freq 等，common_fields 作为表名的一部分
        返回值：
            符合条件的数据列表，类型为 pd.DataFrame。
        """
        table_name = self._get_table_name(common_fields=common_fields)

        if not self.tables.get(table_name): self.tables[table_name] = self._get_table_info(common_fields=common_fields)
        if not self.tables.get(table_name):
            return []

        primary_keys = self._get_primary_keys(common_fields=common_fields)

        cur = self._get_cursor()
        cur.execute(f"""
            SELECT {", ".join(primary_keys)} FROM {table_name};
        """)
        return cur.fetchall()
    
    def select_by_primary_keys(self, keys: List[Dict[str, Any]], common_fields: Fields = {}) -> List[Any]:
        """
        根据主键列表查询缓存的数据条目。

        参数：
            keys: 主键列表，每个主键为一个字典，包含主键字段及其对应的值。
            common_fields: 公共字段，如 freq 等，common_fields 作为表名的一部分
        返回值：
            符合条件的数据列表。
        """
        table_name = self._get_table_name(common_fields=common_fields)

        if not self.tables.get(table_name): self.tables[table_name] = self._get_table_info(common_fields=common_fields)
        if not self.tables.get(table_name):
            return []

        primary_keys = self._get_primary_keys(common_fields=common_fields)
        for each in primary_keys:
            assert each in keys[0].keys(), f"主键字段 {each} 不在提供的 keys 中。"

        all_values = []
        for each in keys:
            value_tuple = ", ".join([str(_python_value_to_sqlite_value(each[pk])) for pk in primary_keys])
            all_values.append(f"({value_tuple})")
        
        sql = f"""
WITH temp_keys({",".join(primary_keys)}) AS ( VALUES {",".join(all_values)} )
SELECT * FROM "{table_name}" JOIN temp_keys USING ({",".join(primary_keys)});
        """
        cur = self._get_cursor()
        cur.execute(sql)
        return cur.fetchall()

    def _connect(self) -> sqlite3.Connection:
        conn = self.connection.get()
        if not isinstance(conn, sqlite3.Connection):
            self.connection.set(
                sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                )
            )
            conn = self.connection.get()
            assert isinstance(conn, sqlite3.Connection), "数据库连接未正确建立。"
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _tx(self):
        """简单事务封装，保证一组操作要么全成功，要么全失败。"""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def _get_cursor(self):
        conn = self._connect()
        return conn.cursor()

    def __del__(self):
        self.close()
    
    def close(self):
        conn = self.connection.get()
        if conn:
            conn.commit()
            conn.close()
            self.connection.set(None)
    
    def _get_table_name(self, common_fields: Fields) -> str:
        hashed_name = sha1(("-".join([str(v) for v in common_fields.values()])).encode()).hexdigest()
        return f"{self.table_basename}_{hashed_name}"
    
    def _get_primary_keys(self, common_fields: Fields) -> List[str]:
        table_name = self._get_table_name(common_fields=common_fields)
        cur = self._get_cursor()
        cur.execute(f"PRAGMA table_info({table_name});")
        primary_keys = [row['name'] for row in cur.fetchall() if row['pk'] == 1]
        return primary_keys

    @abstractmethod
    def _set_table_info(self, data: Any, common_fields: Fields) -> str | Dict[str, str] | bool:
        raise NotImplementedError

    @abstractmethod
    def _get_table_info(self, common_fields: Fields) -> str | Dict[str, str] | bool:
        raise NotImplementedError


__all__ = ["BaseDB", "Fields"]
