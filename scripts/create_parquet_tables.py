import os
from collections.abc import Callable, Sequence
from collections import namedtuple
from matplotlib import artist
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import psycopg

from jazz_graph.etl.transforms import map_array
from jazz_graph.etl.transforms import map_by_index

class CreateNodeData:
    """Handle extraction and transformation of node data."""
    def __init__(self, sql, params: dict, transforms: list[Callable[[pd.DataFrame], pd.DataFrame]]):
        self.sql = sql
        self.params = params
        self.transforms = transforms

    def extract(self, connection) -> pd.DataFrame:
        """Return data from query."""
        return  pd.read_sql(self.sql, connection, params=self.params)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transforms to data."""
        for transform in self.transforms:
            df = transform(df)
        return df

    def extract_transform(self, connection) -> pd.DataFrame:
        """Convenience method: extracts data and applies transforms."""
        return self.transform(self.extract(connection))


SQL = namedtuple('SQL', [
    'artist_sql',
    'song_sql',
    'performance_sql',
    'performance_song_sql',
    'performance_artist_sql',
    'song_artist_sql'
])

def queries() -> SQL:
    """Return the queries needed for data loading."""
    return SQL(
        artist_sql = """
            WITH relevant_jazz AS (
                SELECT
                    recording_id
                FROM jazz_recordings
                WHERE jazz_recordings.release_date >= %(start)s
                    AND jazz_recordings.release_date < %(end)s
            )
                SELECT composer_id as artist_id
                FROM compositions
                JOIN relevant_jazz ON relevant_jazz.recording_id = compositions.recording_id
            UNION
                SELECT artist_id
                FROM
                relevant_jazz
                JOIN recording_to_performer ON recording_to_performer.recording_id = relevant_jazz.recording_id
            ;
        """,
        song_sql = """
            SELECT
                work_id
            FROM compositions
            JOIN jazz_recordings ON compositions.recording_id = jazz_recordings.recording_id
            WHERE jazz_recordings.release_date >= %(start)s
                AND jazz_recordings.release_date < %(end)s
            GROUP BY
                work_id;
        """,
        performance_sql = """
            SELECT
                recording_id,
                discogs_id as discogs_id,
                release_date
            FROM jazz_recordings
            WHERE jazz_recordings.release_date >= %(start)s
                AND jazz_recordings.release_date < %(end)s
        """,
        performance_song_sql = """
            SELECT
                jazz_recordings.recording_id,
                compositions.work_id
            FROM jazz_recordings
            JOIN compositions ON compositions.recording_id = jazz_recordings.recording_id
            WHERE jazz_recordings.release_date >= %(start)s
                AND jazz_recordings.release_date < %(end)s
        """,
        performance_artist_sql = """
            SELECT
                jazz_recordings.recording_id,
                recording_to_performer.artist_id
            FROM
                jazz_recordings
            JOIN
                recording_to_performer ON jazz_recordings.recording_id = recording_to_performer.recording_id
            WHERE jazz_recordings.release_date >= %(start)s
                AND jazz_recordings.release_date < %(end)s
        """,
        song_artist_sql = """
            SELECT
                composer_id as artist,
                work_id as song
            FROM compositions
            JOIN jazz_recordings ON jazz_recordings.recording_id = compositions.recording_id
            WHERE jazz_recordings.release_date >= %(start)s
                AND jazz_recordings.release_date < %(end)s
        """)

def create_tables(start, end, directory):
    """This function extracts data from SQL, transforms to node arrays and edge arrays,
    as parquet tables.
    """
    sql = queries()
    params = {'start': start, 'end': end}
    def merge_labels(performance_nodes:pd.DataFrame) -> pd.DataFrame:
        """Merges the styles data to perfromance nodes."""
        performance_labels = pd.read_parquet('/workspace/local_data/discogs_styles.parquet')
        return performance_nodes.merge(performance_labels, left_on='discogs_id', right_index=True, how='left')

    create_artist_nodes = CreateNodeData(
        sql.artist_sql, params, [lambda df: df.set_index('artist_id')]
    )
    create_song_nodes = CreateNodeData(sql.song_sql, params, [lambda df: df.set_index('work_id')])
    create_performance_nodes = CreateNodeData(
        sql.performance_sql, params,
        [lambda df: df.set_index('recording_id'), merge_labels, lambda df: df.drop(columns=['discogs_id'])])

    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as connection:
        artist_data = create_artist_nodes.extract_transform(connection)
        song_data = create_song_nodes.extract_transform(connection)
        performance_data = create_performance_nodes.extract_transform(connection)

        artist_lookup = map_by_index(artist_data.index)
        song_lookup = map_by_index(song_data.index)
        performance_lookup = map_by_index(performance_data.index)

        performance_artist_data = CreateNodeData(
            sql.performance_artist_sql,
            params,
            [
                lambda df: df.assign(
                    artist_id=map_array(df.artist_id, artist_lookup),
                    recording_id=map_array(df.recording_id, performance_lookup))
            ]
        ).extract_transform(connection)

        performance_song_data = CreateNodeData(
            sql.performance_song_sql,
            params,
            [lambda df: df.assign(
                recording_id=map_array(df.recording_id, performance_lookup),
                work_id=map_array(df.work_id, song_lookup)
            )]
        ).extract_transform(connection)

        song_artist_data = CreateNodeData(
            sql.song_artist_sql,
            params,
            [lambda df: df.assign(
                song=map_array(df.song, song_lookup),
                artist=map_array(df.artist, artist_lookup)
            )]
        ).extract_transform(connection)

    # TODO: stricter validations, e.g., dtype.
    # but map_array will produce nan values, so at least check this.
    # Failure implies something *very* unexpected in source data.
    assert not artist_data.isna().sum().sum(), "Data should not contain any nan values."
    assert not song_data.isna().sum().sum(), "Data should not contain any nan values."
    assert not performance_data.isna().sum().sum(), "Data should not contain any nan values."
    assert not performance_artist_data.isna().sum().sum(), "Data should not contain any nan values."
    assert not performance_song_data.isna().sum().sum(), "Data should not contain any nan values."
    assert not song_artist_data.isna().sum().sum(), f"Data should not contain any nan values in artist_song edges. Found {song_artist_data.isna().sum().sum()} in {len(song_artist_data)} records."

    _write_parquet(artist_data, 'artist_nodes', directory)
    _write_parquet(song_data, 'song_nodes', directory)
    _write_parquet(performance_data, 'performance_nodes', directory)
    _write_parquet(performance_artist_data, 'performance_artist_edges', directory)
    _write_parquet(song_artist_data, 'song_artist_edges', directory)
    _write_parquet(performance_song_data, 'performance_song_edges', directory)


def _read_sql(connection, sql: str) -> pd.DataFrame:
    cur = connection.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    return df

def _write_parquet(df, filename: str, directory: str):
    if not filename.endswith('.parquet'):
        filename = filename + '.parquet'
    path = os.path.join(directory, filename)
    df.to_parquet(path, index=True)
    print(f"Wrote {df.shape[0]} rows, {df.shape[1]} columns to {path}.")

def _read_parquet(filename, directory) -> pd.DataFrame:
    path = os.path.join(directory, filename)
    return pd.read_parquet(path)


if __name__ == '__main__':
    os.makedirs('/workspace/local_data/graph_parquet', exist_ok=True)
    os.makedirs('/workspace/local_data/graph_parquet_proto', exist_ok=True)
    assert os.path.exists('/workspace/local_data/graph_parquet')

    prototype_params = (pd.Timestamp('1957-01-01'), pd.Timestamp('1963-01-01'), '/workspace/local_data/graph_parquet_proto')
    create_tables(*prototype_params)

    production_params = (pd.Timestamp('1900-01-01'), pd.Timestamp('2100-01-01'), '/workspace/local_data/graph_parquet')
    create_tables(*production_params)