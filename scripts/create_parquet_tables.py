import os
from collections.abc import Callable, Sequence
from matplotlib import artist
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import psycopg


class CreateNodeData:
    """Handle extraction and transformation of node data."""
    def __init__(self, sql, transforms: list[Callable[[pd.DataFrame], pd.DataFrame]]):
        self.sql = sql
        self.transforms = transforms

    def extract(self, connection) -> pd.DataFrame:
        return  _read_sql(connection, self.sql)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for transform in self.transforms:
            df = transform(df)
        return df

    def extract_transform(self, connection) -> pd.DataFrame:
        return self.transform(self.extract(connection))


def map_array(arr: ArrayLike, mapping: dict):
    """Map the values in a 1d-array."""
    series = pd.Series(arr)  # pyright: ignore [reportCallIssue, reportArgumentType]
    return series.map(mapping).values

def map_by_index(arr):
    """Create a dictionary mapping values in arr to their index in arr.

    Raises a value error if elements of arr are not unique.
    """
    out = {}
    for i, v in enumerate(arr):
        if v in out:
            raise ValueError("The values in arr must be unique.")
        out[v] = i
    return out

def create_tables():
    arist_sql = """
        SELECT
            artist_id
        FROM jazz_recordings
        JOIN recording_to_performer ON recording_to_performer.recording_id = jazz_recordings.recording
        GROUP BY artist_id
        UNION
        SELECT composer_id as artist_id
        FROM compositions
        GROUP BY composer_id;
    """
    song_sql = """
        SELECT
            work_id
        FROM compositions
        JOIN jazz_recordings ON compositions.recording_id = jazz_recordings.recording
        GROUP BY
            work_id;
    """
    performance_sql = """
        SELECT
            recording,
            MIN(discogs_release) as discogs_release
        FROM jazz_recordings
        GROUP BY recording
    """
    performance_song_sql = """
        SELECT
            jazz_recordings.recording,
            compositions.work_id
        FROM jazz_recordings
        JOIN compositions ON compositions.recording_id = jazz_recordings.recording
        GROUP BY (jazz_recordings.recording, compositions.work_id)
    """
    performance_artist_sql = """
        SELECT
            jazz_recordings.recording,
            recording_to_performer.artist_id
        FROM
            jazz_recordings
        JOIN
            recording_to_performer ON jazz_recordings.recording = recording_to_performer.recording_id
    """
    song_artist_sql = """
        SELECT
            composer_id as artist,
            work_id as song
        FROM compositions
        JOIN jazz_recordings ON jazz_recordings.recording = compositions.recording_id
        GROUP BY (composer_id, work_id)
    """
    def merge_labels(performance_nodes:pd.DataFrame) -> pd.DataFrame:
        performance_labels = pd.read_parquet('/workspace/local_data/discogs_styles.parquet')
        return performance_nodes.merge(performance_labels, left_on='discogs_release', right_index=True, how='left')

    create_artist_nodes = CreateNodeData(
        arist_sql, [lambda df: df.set_index('artist_id')]
    )
    create_song_nodes = CreateNodeData(song_sql, [lambda df: df.set_index('work_id')])
    create_performance_nodes = CreateNodeData(
        performance_sql,
        [lambda df: df.set_index('recording'), merge_labels, lambda df: df.drop(columns=['discogs_release'])])

    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as connection:
        artist_data = create_artist_nodes.extract_transform(connection)
        song_data = create_song_nodes.extract_transform(connection)
        performance_data = create_performance_nodes.extract_transform(connection)

        artist_lookup = map_by_index(artist_data.index)
        song_lookup = map_by_index(song_data.index)
        performance_lookup = map_by_index(performance_data.index)

        performance_artist_data = CreateNodeData(
            performance_artist_sql,
            [
                lambda df: df.assign(
                    artist_id=map_array(df.artist_id, artist_lookup),
                    recording=map_array(df.recording, performance_lookup))
            ]
        ).extract_transform(connection)

        performance_song_data = CreateNodeData(
            performance_song_sql,
            [lambda df: df.assign(
                recording=map_array(df.recording, performance_lookup),
                work_id=map_array(df.work_id, song_lookup)
            )]
        ).extract_transform(connection)

        song_artist_data = CreateNodeData(
            song_artist_sql,
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
    assert not song_artist_data.isna().sum().sum(), "Data should not contain any nan values in artist_song edges."

    _write_parquet(artist_data, 'artist_nodes.parquet')
    _write_parquet(song_data, 'song_nodes.parquet')
    _write_parquet(performance_data, 'performance_nodes.parquet')
    _write_parquet(performance_artist_data, 'performance_artist_edges.parquet')
    _write_parquet(song_artist_data, 'song_artist_edges.parquet')
    _write_parquet(performance_song_data, 'performance_song_edges')


def _read_sql(connection, sql: str) -> pd.DataFrame:
    cur = connection.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    return df

def _write_parquet(df, filename: str, directory: str = '/workspace/local_data'):
    path = os.path.join(directory, filename)
    df.to_parquet(path, index=True)
    print(f"Wrote {df.shape[0]} rows, {df.shape[1]} columns to {path}.")

def _read_parquet(filename, directory) -> pd.DataFrame:
    path = os.path.join(directory, filename)
    return pd.read_parquet(path)



def create_artist_nodes(connection):
    sql = """
    SELECT
        recording_to_performer.artist_id
    FROM jazz_recordings
    JOIN recording_to_performer ON recording_to_performer.recording_id = jazz_recordings.recording
    GROUP BY artist_id;
    """
    df = _read_sql(connection, sql, )

    ...
    _write_parquet(df, 'artist_nodes.parquet')

def create_song_nodes(connection):
    sql = """
    SELECT
        work_id
    FROM compositions
    JOIN jazz_recordings ON compositions.recording_id = jazz_recordings.recording
    GROUP BY
        work_id;
    """
    df = _read_sql(connection, sql)

    _write_parquet(df, 'song_nodes.parquet')

def create_performance_nodes(connection):
    sql = """
    SELECT
        recording
    FROM jazz_recordings
    GROUP BY recording
    """
    _read_sql(connection, sql)

def create_performance_artist_edges(connection):
    sql = """
    SELECT
        jazz_recordings.recording,
        recording_to_performer.artist_id
    FROM
        jazz_recordings
    JOIN
        recording_to_performer ON jazz_recordings.recording = recording_to_performer.recording_id
    """
    _read_sql(connection, sql)

def create_performance_song_edges(connection):
    sql = """
    SELECT
        jazz_recordings.recording,
        compositions.work_id
    FROM jazz_recordings
    JOIN compositions ON compositions.recording_id = jazz_recordings.recording
    GROUP BY (jazz_recordings.recording, compositions.work_id)
    """
    _read_sql(connection, sql)

def create_song_artist_edges(connection):
    sql = """
    SELECT
        composer_id as artist,
        work_id as song
    FROM compositions
    JOIN jazz_recordings ON jazz_recordings.recording = compositions.recording_id
    GROUP BY (composer_id, work_id)
    """
    _read_sql(connection, sql)

if __name__ == '__main__':
    create_tables()