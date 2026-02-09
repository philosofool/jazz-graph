"""Create a table in musicbrainz db with Jazz recordings in it."""
# TODO: This is pretty messy and not normalized.
# It would be better to:
# Write a junction table (recording_id, discog_id) and a table discogs_releases.
# This is just standard normalization practice and a fairly obvious step.
# The intermediate table that is created here should be a materialized view
# of relevant joins.

import psycopg
import pandas as pd

from jazz_graph.etl.extract_discogs import InMemDiscogs, is_jazz_album, MatchDiscogs
from jazz_graph.etl.load import LoadData
from jazz_graph.schema.sql import Column, ForeignKey, PrimaryKey, TableSchema
from jazz_graph.clean.string_date import date_precision, clean_string_date


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
        styles = matching['styles']
        recording_id = row[3]
        if recording_id in self.seen:
            return # this recording already exists in the data, so ignore it.
        self.seen.add(recording_id)
        data = row + (discogs_album_id, release_year, styles)
        self.add_data(data)

    def add_data(self, data):
        self.data.append(data)

    def dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame.from_records(
            self.data,
            columns=['recording_id', 'album_id', 'song', 'album', 'artist', 'release_year', 'release_month', 'release_day', 'discog_album_id', 'discog_date', 'styles']
        )
        return df

def read_text_file(path):
    with open(path) as f:
        out = f.read()
    return out

class Schemas:
    def __init__(self):
        self.discogs_releases = TableSchema(
            'discogs_release',
            [
                Column('id', 'INT', nullable=False, is_identity=False),
                Column('title', 'TEXT'),
                Column('release_date', 'DATE'),
                Column('date_precision', 'TEXT')
            ],
            PrimaryKey(['id'], 'discogs_release_pk', False),
            []
        )
        self.styles = TableSchema(
            'styles',
            [Column('id', 'INT', False, is_identity=True), Column('style_name', 'TEXT', False)],
            PrimaryKey(['id'], 'styles_pk', True),
            []
        )
        self.styles_to_discogs = TableSchema(
            'styles_to_discogs',
            [Column('style_id', 'INT', False), Column('discog_id', 'INT')],
            PrimaryKey(['style_id', 'discogs_id']),
            [ForeignKey(['discogs_id'], ['id'], 'discogs_release')]
        )
        self.recordings_to_discogs = TableSchema(
            'discogs_release_to_recording',
            [
                Column('recording_id', 'INT', False),
                Column('discogs_id', 'INT', False)
            ],
            PrimaryKey(['recording_id', 'discogs_id'], auto_generate=False),
            [
                ForeignKey(['recording_id'], ['id'], 'recording'),
                ForeignKey(['discogs_id'], ['id'], 'discogs_release')
            ]
        )


def create_jazz_data() -> pd.DataFrame:
    # query = read_text_file('/workspace/queries/recording_to_album.sql')
    query = "SELECT * FROM recording_to_album;"
    discogs = MatchDiscogs(InMemDiscogs('/workspace/local_data/jazz_releases.jsonl', is_jazz_album))
    process_rows = ProcessRows()
    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:
        conn.autocommit = False  # REQUIRED

        cur = conn.cursor(name="my_streaming_cursor")  # named cursor
        cur.itersize = 10_000

        cur.execute(query)   # pyright: ignore [reportArgumentType]
        n_row_processed = 0
        n_row_inspected = 0
        for row in cur:
            n_row_inspected += 1
            matched = discogs.matching_discog(row)
            if not matched:
                continue
            process_rows.process_matched(row, matched)
            n_row_processed += 1
        cur.close()
        print(f"Finished processing query.\nRows seen: {n_row_inspected}.\nProcessed {n_row_processed} jazz entries with {len(process_rows.data)} unique recordings.")
    return process_rows.dataframe()


def create_jazz_table(discogs_data: pd.DataFrame, schema: TableSchema):
    # schema = Schemas().discogs_releases
    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:

        cursor = conn.cursor()
        load_data = LoadData(schema)
        load_data.create_table(cursor, True)
        load_data.load_data(discogs_data, cursor)
        cursor.close()

        cursor = conn.cursor(name="my_streaming_cursor")
        cursor.execute(f"SELECT * FROM {schema.name} LIMIT 3")
        for row in cursor:
            print(row)
        cursor.close()


def data_to_discogs_schema(df: pd.DataFrame):
    """Transform jazz_data output for loading according to discogs_release schema."""
    selected_columns = ['discog_album_id', 'song', 'discog_date']
    df = df[selected_columns]
    df['date_precision'] = date_precision(df['discog_date'])
    df['discog_date'] = clean_string_date(df['discog_date'])
    return df

def data_to_recording_discog(jazz_data: pd.DataFrame):
    jazz_data = jazz_data[['recording_id', 'discog_album_id']]
    print(f"Dropping {jazz_data.duplicated().sum()} duplicated entries.")
    jazz_data = jazz_data.drop_duplicates()
    return jazz_data


if __name__ == '__main__':
    jazz_data_path = '/workspace/local_data/jazz_data_discogs.csv'
    try:
        jazz_data = pd.read_csv(jazz_data_path)
        print("Loaded cached csv of jazz data.")
    except FileNotFoundError:
        print("No cached data. Creating jazz data and caching.")
        jazz_data = create_jazz_data()
        jazz_data.to_csv(jazz_data_path, index=False)

    print(jazz_data.head())
    jazz_styles = jazz_data.styles.explode()
    jazz_styles = pd.DataFrame(jazz_styles, index=jazz_styles.index, columns=['styles'])
    jazz_styles = jazz_styles.merge(jazz_data.discog_album_id, left_index=True, right_index=True, how='left')
    print(jazz_sytles.head())


    # create styles in db:

    # TODO: Maybe--load styles to DB.
    # jazz_styles_data = pd.DataFrame([jazz_styles.styles.unique()], columns=['styles'])
    # styles_loader = LoadData(Schemas().styles)

    discogs_jazz = data_to_discogs_schema(jazz_data)
    load_discogs = LoadData(Schemas().discogs_releases)

    # TODO: Maybe--create_styles_to_discogs
    # TODO: double check column order.
    # LoadData(Schema().styles_to_discogs).load_data(jazz_styles_data)

    recording_to_discog_data = data_to_recording_discog(jazz_data)
    load_recording_to_discog = LoadData(Schemas().recordings_to_discogs)

    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:
        conn.autocommit = False
        cursor = conn.cursor()
        # create_discogs_jazz

        load_discogs.create_table(cursor)
        load_discogs.load_data(discogs_jazz, cursor)

        load_recording_to_discog.create_table(cursor)
        load_recording_to_discog.load_data(recording_to_discog_data, cursor)
        conn.commit()


    #     # create recordings_to_discogs
    #     # ...filter data to correct columns.
    #     LoadData(Schema().recordings_to_discogs).load_data(...)
