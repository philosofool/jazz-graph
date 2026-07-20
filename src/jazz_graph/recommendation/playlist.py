"""Handle incoming data for recommendation."""
from collections.abc import Iterable, Iterator
import pandas as pd
import numpy as np
from jazz_graph.clean.data_normalization import normalize_title

class ListenField:
    def __init__(self, field_name, field_key):
        self.field_name = field_name
        self.field_key = field_key

class ListenSchema:
    def __init__(self, fields: list[ListenField]):
        self._fields = fields

    def fields(self) -> list:
        return [f.field_key for f in self._fields]

    def make_key(self, **kwargs):
        out = list()
        for field in self._fields:
            if field.field_name in kwargs:
                out.append(kwargs[field.field_name])
        return tuple(out)


class SpotifyListens:
    def __init__(self, recording_traits: pd.DataFrame, schema: ListenSchema | None = None):
        if schema is None:
            self.schema = ListenSchema([
                ListenField('title', 'master_metadata_track_name'),
                ListenField('album', 'master_metadata_album_album_name'),
                ListenField('artist', 'master_metadata_album_artist_name'),
            ])
        else:
            self.schema = schema
        self.recording_traits = recording_traits.sort_values(['release_date'])
        self.lookup: dict[tuple, int] = {}

        for row in recording_traits.itertuples():
            recording_id = row.Index
            title = normalize_title(row.title)
            album = normalize_title(row.album)
            artist = normalize_title(row.artist)
            key = self.schema.make_key(title=title, artist=artist, album=album)
            if key not in self.lookup:
                self.lookup[key] = recording_id    # pyright: ignore [reportArgumentType]

    def get_recording_id(self, record: dict) -> int|None:
        norm_key = tuple()
        for field in self.schema.fields():
            data = record.get(field)
            if data is None:
                return None
            norm_key += (normalize_title(data),)

        return self.lookup.get(norm_key)

    def get_spotify_jazz(self, spotify_data: list[dict], unique=True) -> Iterable[tuple[dict, int]]:
        """Return a generator for jazz records in spotify data."""
        yield from self._yield_spotify_matches(spotify_data, unique, False)

    def get_spotify_misses(self, spotify_data: list[dict], unique=True) -> Iterable[tuple[dict, int]]:
        """Return a generator for missing jazz records in spotify data."""
        yield from self._yield_spotify_matches(spotify_data, unique, True)

    def _get_spotify_id(self, record: dict):
        if 'spotify_track_uri' in record:
            return  record.get('spotify_track_uri')
        out = []
        for field in self.schema.fields():
            item = record.get(field)
            out.append(item)
        return tuple(out)

    def _yield_spotify_matches(self, spotify_data: list[dict], unique, invert) -> Iterable[tuple[dict, int]]:
        seen = set()
        for record in spotify_data:
            spot_id = self._get_spotify_id(record)
            if unique and spot_id in seen:
                continue
            if spot_id is None:
                continue
            seen.add(spot_id)
            recording_id = self.get_recording_id(record)
            if recording_id is None:
                if invert:
                    yield record, None
                continue
            elif not invert:
                yield record, recording_id

    def get_listen_ids(self, spotify_data: list[dict], unique=True) -> np.ndarray:
        iterator = (rec_id  for _, rec_id in self.get_spotify_jazz(spotify_data, unique))
        return np.fromiter(iterator, dtype=np.int64)

    def get_listen_data(self, spotify_data: list[dict], unique=True) -> Iterator:
        return self.recording_traits.loc[self.get_listen_ids(spotify_data)].itertuples()
