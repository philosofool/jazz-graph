from jazz_graph.schema.sql import Column, ForeignKey, PrimaryKey, TableSchema
import re

def test_column():
    column = Column('col', 'INT')
    assert column.nullable is True
    assert column.name == 'col'
    assert column.dtype == 'INT'

class TestTableSchema:
    def test_create_table_sql(self):
        table_schema = TableSchema(
            'the_table',
            [Column('pk', 'INT'), Column('col_b', 'INT'), Column('col_c', 'BOOLEAN')],
            PrimaryKey(['pk']),
            []
        )
        sql = table_schema.create_table_sql()
        assert 'pk INT,\n' in sql
        assert 'col_b INT,\n' in sql
        assert sql.endswith(');\n')
        starts_with_create_table = re.match(r'\s*CREATE TABLE the_table\s+\(', sql)
        assert starts_with_create_table is not None, f"Expected to start with valid CREATE TABLE statment, got {sql}"
        assert table_schema._primary_key_sql() in sql

    def test_create_table_with_foreign_key(self):
        table_schema = TableSchema(
            'the_table',
            [Column('pk', 'INT'), Column('col_b', 'INT'), Column('col_c', 'BOOLEAN')],
            PrimaryKey(['pk']),
            [ForeignKey(['col_b'], ['id'], 'other_table')]
        )
        sql = table_schema.create_table_sql()
        assert table_schema._foreign_key_sql() in sql

    def test_copy_sql(self):
        table_schema = TableSchema(
            'the_table',
            [Column('pk', 'INT'), Column('col_b', 'INT'), Column('col_c', 'BOOLEAN')],
            PrimaryKey(['pk']),
            []
        )
        sql = table_schema.copy_data_sql()
        expected = r'\s*COPY the_table \('
        assert re.match(expected, sql)
        assert 'pk, col_b, col_c' in sql
        assert 'FROM STDIN' in sql
        assert 'WITH (FORMAT CSV, HEADER)' in sql

        table_schema = TableSchema(
            'the_table',
            [Column('pk', 'INT', is_identity=True), Column('col_b', 'INT'), Column('col_c', 'BOOLEAN')],
            PrimaryKey(['pk']),
            []
        )
        assert 'pk' not in table_schema.copy_data_sql(), "Identies should not be included in copy statement."

    def test_drop_table_sql(self):
        table_schema = TableSchema(
            'the_table',
            [Column('pk', 'INT'), Column('col_b', 'INT'), Column('col_c', 'BOOLEAN')],
            PrimaryKey(['pk']),
            []
        )
        sql = table_schema.drop_table_sql(if_exists=True)
        assert sql == 'DROP TABLE IF EXISTS the_table;'
        sql = table_schema.drop_table_sql(if_exists=False)
        assert sql == "DROP TABLE the_table;"
