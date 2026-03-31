"""Handle incoming data for recommendation."""
from collections.abc import Iterable, Iterator
import pandas as pd
import numpy as np
from jazz_graph.clean.data_normalization import normalize_title


class SpotifyListens:
    def __init__(self, recording_traits: pd.DataFrame):
        self.recording_traits = recording_traits.sort_values(['release_date'])
        self.lookup: dict[tuple, int] = {}
        for row in recording_traits.itertuples():
            recording_id = row.Index
            title = normalize_title(row.title)
            album = normalize_title(row.album)
            artist = normalize_title(row.artist)
            key = (title, album, artist)
            if key not in self.lookup:
                self.lookup[key] = recording_id    # pyright: ignore [reportArgumentType]

    def get_recording_id(self, record: dict) -> int|None:
        fields = 'master_metadata_track_name', 'master_metadata_album_album_name', 'master_metadata_album_artist_name'
        norm_key = tuple(normalize_title(record[field]) for field in fields)
        return self.lookup.get(norm_key)

    def get_spotify_jazz(self, spotify_data: list[dict], unique=True) -> Iterable[tuple[dict, int]]:
        """Return a generator for jazz records in spotify data."""
        yield from self._yield_spotify_matches(spotify_data, unique, False)

    def get_spotify_misses(self, spotify_data: list[dict], unique=True) -> Iterable[tuple[dict, int]]:
        """Return a generator for jazz records in spotify data."""
        yield from self._yield_spotify_matches(spotify_data, unique, True)

        # seen = set()
        # for record in spotify_data:
        #     spot_id = record['spotify_track_uri']
        #     if unique and spot_id in seen:
        #         continue
        #     seen.add(spot_id)
        #     recording_id = self.get_recording_id(record)
        #     if recording_id is None:
        #         continue
        #     yield record, recording_id

    def _yield_spotify_matches(self, spotify_data: list[dict], unique, invert) -> Iterable[tuple[dict, int]]:
        seen = set()
        for record in spotify_data:
            spot_id = record['spotify_track_uri']
            if unique and spot_id in seen:
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
