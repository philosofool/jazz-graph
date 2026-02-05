from functools import lru_cache
import io

import psycopg
import pandas as pd

from jazz_graph.extract_discogs import InMemDiscogs, is_jazz_album, MatchDiscogs


class ProcessRows:
    def __init__(self):
        self.seen = set()
        self.data = []

    def process_matched(self, row, matching):
        # This processes the data so that we get one row per jazz recording.
        # Note the the most important data is the recording id and discog id.
        # The discog id will allow us to determine the most appropriate master
        # album to associate with the recording. BUT we're not yet concerned
        # with albums. The main data model is performance, musician, song.

        # row = (song, album, artist, recording_id, album_id, release_year, release_month, release_day)
        discogs_album_id = matching['id']
        release_year = matching['released']
        recording_id = row[3]
        if recording_id in self.seen:
            return # this recording already exists in the data, so ignore it.
        self.seen.add(recording_id)
        data = row + (discogs_album_id, release_year)
        self.add_data(data)

    def add_data(self, data):
        self.data.append(data)

    def data_csv_stream(self) -> io.StringIO:
        df = pd.DataFrame.from_records(
            self.data,
            columns=['song', 'album', 'artist', 'recording_id', 'album_id', 'release_year', 'release_month', 'release_day', 'discog_album_id', 'discog_date']
        )[['recording_id', 'album_id', 'song', 'album', 'artist', 'discog_album_id']]
        csv_buff = io.StringIO()
        df.to_csv(csv_buff, index=False)
        csv_buff.seek(0)
        return csv_buff

class LoadData():
    """Load the data."""
    def __init__(self, connection):
        self.connection = connection

    def load_data(self, csv_buff):
        try:
            # to a table in the database.
            cursor = self.connection.cursor()
            self._create_jazz_recordings_table(cursor)
            # song, album, artist, recording_id, album_id, release_year, release_month, release_day, discog_id, discog_date
            with cursor.copy("""
                COPY jazz_recordings (
                    recording,
                    record_group,
                    song_title,
                    album_title,
                    artist_name,
                    discogs_release
                )
                FROM STDIN
                WITH (FORMAT CSV, HEADER)
            """) as copy:
                copy.write(csv_buff.read())
            cursor.close()
            self.connection.commit()
        except Exception:
            self.connection.rollback()

    def _create_jazz_recordings_table(self, cursor):
        import textwrap
        cursor.execute("DROP TABLE IF EXISTS jazz_recordings;")
        query = textwrap.dedent("""
        CREATE TABLE jazz_recordings (
            id SERIAL PRIMARY KEY,
            recording INT NOT NULL,
            record_group INT,
            song_title TEXT,
            album_title TEXT,
            artist_name TEXT,
            discogs_release INT,
            CONSTRAINT fk_recording
                FOREIGN KEY (recording)
                REFERENCES recording(id)
                ON DELETE RESTRICT
        );
        """)
        cursor.execute(query)

class Pipeline:
    def __init__(self, path_to_query):
        self.query = self.import_query(path_to_query)
        self.discogs = MatchDiscogs(InMemDiscogs('/workspace/local_data/jazz_releases.jsonl', is_jazz_album))
        self.process_rows = ProcessRows()

    def execute(self):
        with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:
            conn.autocommit = False  # REQUIRED

            cur = conn.cursor(name="my_streaming_cursor")  # named cursor
            cur.itersize = 10_000

            cur.execute(self.query)
            n_row_processed = 0
            n_row_inspected = 0
            for row in cur:
                n_row_inspected += 1
                matched = self.discogs.matching_discog(row)
                if not matched:
                    continue
                self.process_rows.process_matched(row, matched)
                n_row_processed += 1

            print(f"Finished processing query.\nRows seen: {n_row_inspected}.\nProcessed {n_row_processed} jazz entries with {len(self.process_rows.data)} unique recordings.")

            load_data = LoadData(conn)
            load_data.load_data(self.process_rows.data_csv_stream())
            cur.execute("SELECT * FROM jazz_recordings LIMIT 3")
            for row in cur:
                print(row)
            cur.close()

    @staticmethod
    def import_query(path):
        with open(path) as f:
            out = f.read()
        return out

if __name__ == '__main__':
    import os
    sql = '/workspace/queries/recording_to_album.sql'
    pipeline = Pipeline(sql)
    pipeline.execute()
