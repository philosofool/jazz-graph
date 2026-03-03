"""Contains functions for fetching often used tables in MusicBrainz."""

import pandas as pd
import psycopg

def fetch_recording_traits(
        start: pd.Timestamp | None =None, end: pd.Timestamp | None =None, use_proto: bool = False) -> pd.DataFrame:
    """Helper function to retrive known jazz recordings in MusicBrainz.

    start:
        Earliest release date to include in returned records. Inclusive
    end:
        Latest release data to include in returned records. Exclusive.
    use_proto:
        Return records from 1957 to 1962 (inclusive.)

    Returns
    -------
    The data in jazz_recorings.

    """
    conn = psycopg.connect("dbname=musicbrainz_db user=philosofool")
    if use_proto:
        assert start is None and end is None, "Start and end should be None if using prototyping data."
        start = pd.Timestamp('1957-01-01')
        end = pd.Timestamp('1963-01-01')
    start = pd.Timestamp(start) if start is not None else pd.Timestamp('1900-01-01')
    end = pd.Timestamp(end) if end is not None else pd.Timestamp('2100-01-01')
    sql = """
    SELECT * FROM jazz_recordings
    WHERE jazz_recordings.release_date >= %(start)s
        AND jazz_recordings.release_date < %(end)s;"""
    return pd.read_sql(sql, conn, params={'start': start, 'end': end})

def fetch_discogs_to_recording_id():
    """Helper funcitnoto retrive id mapping from discogs to recording."""
    conn = psycopg.connect("dbname=musicbrainz_db user=philosofool")
    sql = "SELECT * FROM discogs_release_to_recording;"
    return pd.read_sql(sql, conn)

def fetch_compositions():
    """Retrives compositions table."""
    conn = psycopg.connect("dbname=musicbrainz_db user=philosofool")
    sql = "SELECT * FROM compositions;"
    return pd.read_sql(sql, conn)