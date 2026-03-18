"""Handle incoming data for recommendation."""
from collections.abc import Iterable
import pandas as pd
import numpy as np
from jazz_graph.clean.data_normalization import normalize_title


class SpotifyListens:
    def __init__(self, data: pd.DataFrame):
        self.lookup: dict[tuple, int] = {}
        for row in data.itertuples():
            row.Index
            title = normalize_title(row.title)
            album = normalize_title(row.album)
            artist = normalize_title(row.artist)
            self.lookup[(title, album, artist)] = row.recording_id    # pyright: ignore [reportArgumentType]

    def get_recording_id(self, record: dict) -> int:
        fields = 'master_metadata_track_name', 'master_metadata_album_album_name', 'master_metadata_album_artist_name'
        norm_key = tuple(normalize_title(record[field]) for field in fields)
        return self.lookup.get(norm_key)    # pyright: ignore [reportReturnType]

    def get_spotify_jazz(self, spotify_data: list[dict], unique=True) -> Iterable[tuple[dict, int]]:
        """Return a generator for jazz records in spotify data."""
        seen = set()
        for record in spotify_data:
            spot_id = record['spotify_track_uri']
            if unique and spot_id in seen:
                continue
            seen.add(spot_id)
            recording_id = self.get_recording_id(record)
            if recording_id is None:
                continue
            yield record, recording_id

    def get_listen_ids(self, spotify_data: list[dict], unique=True) -> np.ndarray:
        ids = []
        for _, rec_id in self.get_spotify_jazz(spotify_data, unique):
            ids.append(rec_id)
        return np.array(ids)