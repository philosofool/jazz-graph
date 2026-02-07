from dataclasses import dataclass
import textwrap


@dataclass
class Column:
    name: str
    dtype: str
    nullable: bool = True
    is_identity: bool = False

@dataclass
class PrimaryKey:
    columns: list[str]
    name: str | None = None
    auto_generate: bool = True

    def primary_key_col_sql(self, column: Column):
        """Generate a create table clause for column as a primary key."""
        ...

    def constraint_sql(self, table: 'TableSchema'):
        if len(self.columns) == 1:
            pkey = self.columns[0]
        elif len(self.columns):
            pkey = f"({','.join([key for key in self.columns])})"
        else:
            raise NotImplementedError("Primary key columns must be non-empty.")
        constraint_name = self.name or f"{table.name}_pkey"
        return f"CONSTRAINT {constraint_name}\n   PRIMARY KEY {pkey}"


@dataclass
class ForeignKey:
    local_columns: list[str]
    references_columns: list[str]
    references_table: str
    on_delete: str = "RESTRICT"

    def constraint_sql(self, table: 'TableSchema') -> str:
        name = '_'.join(self.local_columns)
        name = name + '_fk'
        ref_cols = ','.join(self.references_columns)
        foreign_key = ', '.join(self.local_columns)
        return textwrap.dedent(f"""
            CONSTRAINT {name}
                FOREIGN KEY ({foreign_key})
                REFERENCES {self.references_table}({ref_cols})
                ON DELETE {self.on_delete}""")

@dataclass
class TableSchema:
    name: str
    columns: list[Column]
    primary_key: PrimaryKey
    foreign_keys: list[ForeignKey]

    def create_table_sql(self) -> str:
        cols_sql = ',\n'.join([f"{col.name} {col.dtype}" for col in self.columns])
        fk_constraints = self._foreign_key_sql()
        sql = f"""
        CREATE TABLE {self.name} (
            {cols_sql},
            {self._primary_key_sql()}
            {fk_constraints}
        );
        """
        return textwrap.dedent(sql)

    def copy_data_sql(self):
        columns = ', '.join([col.name for col in self.columns if not col.is_identity])
        return f"""
            COPY {self.name} (
                {columns}
            )
            FROM STDIN
            WITH (FORMAT CSV, HEADER)
        """

    def drop_table_sql(self, if_exists=True):
        if not if_exists:
            return f"DROP TABLE {self.name};"
        return f"DROP TABLE IF EXISTS {self.name};"

    def _primary_key_sql(self):
        return self.primary_key.constraint_sql(self)

    def _foreign_key_sql(self):
        return '\n'.join([fk.constraint_sql(self) for fk in self.foreign_keys])
