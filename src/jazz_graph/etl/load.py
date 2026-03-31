from __future__ import annotations
from typing import TYPE_CHECKING
from jazz_graph.data.schema.sql import TableSchema
import io

if TYPE_CHECKING:
    import pandas as pd


class LoadData:
    """Load data based, using the table schema.

    The table schema determines the correct sql statement
    for the table loading, table creation (if requested),
    and handles rebuilds if necessary.
    """
    def __init__(self, schema: TableSchema):
        self.schema = schema

    def load_data(self, df: pd.DataFrame, cursor):
        csv_buff = self.pandas_to_stream(df)
        sql = self.schema.copy_data_sql()
        with cursor.copy(sql) as copy:
            copy.write(csv_buff.read())

    def create_table(self, cursor, drop_if_exists: bool = False):
        if drop_if_exists:
            cursor.execute(self.schema.drop_table_sql())
        sql = self.schema.create_table_sql()
        cursor.execute(sql)

    def pandas_to_stream(self, df: pd.DataFrame):
        csv_buff = io.StringIO()
        df.to_csv(csv_buff, index=False)
        csv_buff.seek(0)
        return csv_buff