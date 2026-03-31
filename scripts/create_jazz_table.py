"""Create a table in musicbrainz db with Jazz recordings in it.

This script adds
    discogs_release
    discogs_release_to_recording
to musicbrainz db.
"""

import psycopg
import pandas as pd

from jazz_graph.etl.extract_discogs import InMemDiscogs, is_jazz_album, MatchDiscogs
from jazz_graph.etl.load import LoadData
from jazz_graph.data.schema.sql import Column, ForeignKey, PrimaryKey, TableSchema
from jazz_graph.clean.string_date import date_precision, clean_string_date
from jazz_graph.clean.data_normalization import normalize_title


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

        # row = (recording_id, release_group_id, song, album, artist, release_date)
        discogs_album_id = matching['id']
        release_year = matching['released']
        styles = matching['styles']
        recording_id = row[0]
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
            columns=['recording_id', 'release_group_id', 'song', 'album', 'artist', 'release_date', 'discog_album_id', 'discog_date', 'styles']
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
                Column('title', 'TEXT'),  # album title, as in Discogs.
                Column('release_date', 'DATE'),  # release date, acc. to discogs.
                Column('date_precision', 'TEXT')  # Dates are the latest possible date compatible with available information.
            ],
            PrimaryKey(['id'], 'discogs_release_pk', False),
            []
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


def create_jazz_data() -> pd.DataFrame:
    """Extract first release data and match against known discogs data."""
    query = "SELECT * FROM recording_first_release;"
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


def data_to_discogs_schema(df: pd.DataFrame):
    """Transform jazz_data output for loading according to discogs_release schema."""
    # TODO: this should be driven by a schema.
    selected_columns = ['discog_album_id', 'album', 'discog_date']
    df = df[selected_columns]
    df['date_precision'] = date_precision(df['discog_date'])
    df['discog_date'] = clean_string_date(df['discog_date'])
    return df.drop_duplicates(subset='discog_album_id')

def data_to_recording_discog(jazz_data: pd.DataFrame):
    """Transform jazz_Data output for loading according to recording_to_discogs schema."""
    # TODO: this should be driven by a schema.
    jazz_data = jazz_data[['recording_id', 'discog_album_id']]
    print(f"Dropping {jazz_data.duplicated().sum()} duplicated entries.")
    jazz_data = jazz_data.drop_duplicates()
    return jazz_data

def normalize_columns(df, columns: list) -> pd.DataFrame:
    """Apply normalize_title to df columns."""
    df = df.copy()
    for column in columns:
        normalized = df[column].apply(normalize_title)
        df[f"{column}_normalized"] = normalized
    return df

def deduplicate(df: pd.DataFrame):
    """Deduplicate records in jazz_data."""
    orig_columns = df.columns.to_list()
    df = normalize_columns(df, ['artist', 'song', 'album'])
    print(df.columns)
    df = (
        df
        .sort_values('release_date')
        .drop_duplicates(
            subset=['artist_normalized', 'song_normalized', 'album_normalized'])
        .drop(columns=['artist_normalized', 'song_normalized', 'album_normalized'])
    )
    return df

def test_deduplicate():
    """This is just a test, but the code in here isn't in scr, so we just test here."""
    df = pd.DataFrame({
        'song': ['One', 'One (5.0 Mix)', 'Two'],
        'album': ['three', 'three', 'four'],
        'artist': ['Foo', 'Foo', 'Bar'],
        'release_date': [1, 0, 2]
    })
    dedup = deduplicate(df)
    assert len(dedup) == 2, "Should drop one duplicated."
    assert 0 in df.release_date.values, "Should keep the entry with the smallest release date."
    assert 'Bar' in dedup.artist.values, "This data should not be one of the dropped rows."
    assert not any(('_norm' in col for col in dedup.columns)), "Should drop normalized columns."



if __name__ == '__main__':
    test_deduplicate()
    jazz_data_path = '/workspace/local_data/jazz_data_discogs.csv'
    try:
        jazz_data = pd.read_csv(jazz_data_path)
        print("Loaded cached csv of jazz data.")
    except FileNotFoundError:
        print("No cached data. Creating jazz data and caching.")
        jazz_data = create_jazz_data()
        jazz_data.to_csv(jazz_data_path, index=False)

    jazz_data = deduplicate(jazz_data)

    # TODO: Maybe--load styles to DB.
    # jazz_styles = jazz_data.styles.explode()
    # jazz_styles = pd.DataFrame(jazz_styles, index=jazz_styles.index, columns=['styles'])
    # jazz_styles = jazz_styles.merge(jazz_data.discog_album_id, left_index=True, right_index=True, how='left')
    # print(jazz_styles.head())
    # jazz_styles_data = pd.DataFrame([jazz_styles.styles.unique()], columns=['styles'])
    # styles_loader = LoadData(Schemas().styles)


    discogs_jazz = data_to_discogs_schema(jazz_data)
    print(discogs_jazz.head(20))
    discogs_jazz_loader = LoadData(Schemas().discogs_releases)

    # TODO: Maybe--create_styles_to_discogs
    # TODO: double check column order.
    # LoadData(Schema().styles_to_discogs).load_data(jazz_styles_data)

    recording_to_discogs_data = data_to_recording_discog(jazz_data)
    recording_to_discog_loader = LoadData(Schemas().recordings_to_discogs)

    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as conn:
        conn.autocommit = False
        cursor = conn.cursor()

        discogs_jazz_loader.create_table(cursor)
        discogs_jazz_loader.load_data(discogs_jazz, cursor)

        recording_to_discog_loader.create_table(cursor)
        recording_to_discog_loader.load_data(recording_to_discogs_data, cursor)
        conn.commit()
