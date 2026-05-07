from jazz_graph.etl.load import LoadData
from jazz_graph.data.schema.sql import TableSchema, PrimaryKey, Column

import pandas as pd
import pytest

@pytest.fixture
def schema() -> TableSchema:
    schema = TableSchema(
        'the_table',
        [Column('pk', 'INT'), Column('col_b', 'INT'), Column('col_c', 'BOOLEAN')],
        PrimaryKey(['pk']),
        []
    )
    return schema

@pytest.fixture
def df() -> pd.DataFrame:
    df = pd.DataFrame({'pk': [1, 2, 3], 'col_b': [4, 6, 5], 'col_c': [15, 12, 14]})
    return df

class MockCopy:
    def __init__(self, sql: str):
        self.sql = sql
        self.buffer = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def write(self, data: str):
        self.buffer += data


class MockCursor:
    def __init__(self):
        self.executed_sql = []
        self.copy_calls = []

    def execute(self, sql: str):
        self.executed_sql.append(sql)

    def copy(self, sql: str):
        copy_obj = MockCopy(sql)
        self.copy_calls.append(copy_obj)
        return copy_obj


class TestLoadData:
    def test_pandas_to_stream(self, schema: TableSchema, df: pd.DataFrame):
        load_data = LoadData(schema)
        string_io = load_data.pandas_to_stream(df)
        data = string_io.read()
        assert data, "Data should not be empty."
        string_io.seek(0)
        data2 = string_io.read()
        assert data == data2, "Data read from start of stream should matcht that data initially returned."

    def test_load_data(self, schema, df):
        mock_cursor = MockCursor()
        load_data = LoadData(schema)

        load_data.load_data(df, mock_cursor)
        copy_obj = mock_cursor.copy_calls.pop()

        assert copy_obj.buffer == load_data.pandas_to_stream(df).read()

    def test_create_table(self, schema: TableSchema, df):
        mock_cursor = MockCursor()
        load_data = LoadData(schema)
        load_data.create_table(mock_cursor, True)

        executed = mock_cursor.executed_sql.pop()
        expected = schema.create_table_sql()
        assert executed == expected, "Table creation SQL should have been executed last.."

        executed = mock_cursor.executed_sql.pop()
        expected = schema.drop_table_sql()
        assert executed == expected, "Table should have been dropped before creation."

        load_data.create_table(mock_cursor, False)
        assert mock_cursor.executed_sql == [schema.create_table_sql()], "The only SQL executed should be the creation SQL."