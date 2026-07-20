"""Contains functions for fetching often used tables in MusicBrainz."""

import pandas as pd
import psycopg

import warnings

def fetch_recording_traits(
        start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, use_proto: bool = False) -> pd.DataFrame:
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
    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:
        return pd.read_sql(sql, conn, params={'start': start, 'end': end})

def fetch_artist_performance_traits(
        start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, use_proto: bool = False
):
    """Maps performer roles to recordings, for example who played what instrument in a performance."""

    # FIXME: needs composers too.
    warnings.warn(
        """fetch_artist_performance_traits does not include composers;
        use caution if implementing core functionality."""
    )
    sql = """
        SELECT
            recording_to_performer.*
        FROM
            jazz_recordings
        JOIN
            recording_to_performer ON jazz_recordings.recording_id = recording_to_performer.recording_id
        WHERE jazz_recordings.release_date >= %(start)s
            AND jazz_recordings.release_date < %(end)s
    """
    if use_proto:
        assert start is None and end is None, "Start and end should be None if using prototyping data."
        start = pd.Timestamp('1957-01-01')
        end = pd.Timestamp('1963-01-01')
    start = pd.Timestamp(start) if start is not None else pd.Timestamp('1900-01-01')
    end = pd.Timestamp(end) if end is not None else pd.Timestamp('2100-01-01')
    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:
        query_result = pd.read_sql(sql, conn, params={'start': start, 'end': end})
    return query_result

def fetch_artist_traits(start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, use_proto: bool = False):
    """Helper function to retrive known jazz artists in MusicBrainz.

    start:
        Earliest release date to include in returned records. Inclusive
    end:
        Latest release data to include in returned records. Exclusive.
    use_proto:
        Return records from 1957 to 1962 (inclusive.)

    Returns
    -------
    The data in jazz_recorings for artists.

    """
    sql =  """
            WITH relevant_jazz AS (
                SELECT
                    recording_id
                FROM jazz_recordings
                WHERE jazz_recordings.release_date >= %(start)s
                    AND jazz_recordings.release_date < %(end)s
            )
                SELECT composer_id as artist_id, composer as artist_name
                FROM compositions
                JOIN relevant_jazz ON relevant_jazz.recording_id = compositions.recording_id
            UNION
                SELECT artist_id, recording_to_performer.artist_name as artist_name
                FROM
                relevant_jazz
                JOIN recording_to_performer ON recording_to_performer.recording_id = relevant_jazz.recording_id
            ;
        """
    if use_proto:
        assert start is None and end is None, "Start and end should be None if using prototyping data."
        start = pd.Timestamp('1957-01-01')
        end = pd.Timestamp('1963-01-01')
    start = pd.Timestamp(start) if start is not None else pd.Timestamp('1900-01-01')
    end = pd.Timestamp(end) if end is not None else pd.Timestamp('2100-01-01')
    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:
        query_result = pd.read_sql(sql, conn, params={'start': start, 'end': end})
    return query_result.set_index('artist_id')
    # artist_traits = query_result.drop_duplicates(subset=['artist_id'])[['artist_id', 'artist_name', 'instrument']]
    # return
    # return query_result
    # artist_recording_traits = fetch_artist_performance_traits(start, end, use_proto)
    # return artist_traits.set_index('artist_id')

def fetch_song_traits(start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, use_proto: bool = False):
    """Helper function to retrive known jazz songs in MusicBrainz.

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
    sql = """
        WITH jazz_compositions AS (
        SELECT DISTINCT
            work_id, song_title
        FROM compositions
        JOIN jazz_recordings ON compositions.recording_id = jazz_recordings.recording_id
        WHERE jazz_recordings.release_date >= %(start)s
            AND jazz_recordings.release_date < %(end)s
        )
        SELECT
            *
        FROM jazz_compositions
--        FROM compositions as comp
--        JOIN jazz_compositions ON comp.work_id = jazz_compositions.work_id
    """
    if use_proto:
        assert start is None and end is None, "Start and end should be None if using prototyping data."
        start = pd.Timestamp('1957-01-01')
        end = pd.Timestamp('1963-01-01')
    start = pd.Timestamp(start) if start is not None else pd.Timestamp('1900-01-01')
    end = pd.Timestamp(end) if end is not None else pd.Timestamp('2100-01-01')
    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:
        query_result = pd.read_sql(sql, conn, params={'start': start, 'end': end})
    return query_result.set_index('work_id')


def fetch_discogs_to_recording_id():
    """Helper funcitnoto retrive id mapping from discogs to recording."""
    conn = psycopg.connect("dbname=musicbrainz_db user=philosofool")
    sql = "SELECT * FROM discogs_release_to_recording;"
    return pd.read_sql(sql, conn)

def fetch_compositions():
    """Retrives compositions table."""

    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:
        sql = "SELECT * FROM compositions;"
        return pd.read_sql(sql, conn)