import pandas as pd
from jazz_graph.recommendation.playlist import SpotifyListens


class TestSpotifyListens:
    test_data = pd.DataFrame.from_records([
        (1000, 11, "Title 1", "Album 1", "Artist 1", '1999-01-07'),
        (1000, 10, "Title 1", "Album 2", "Artist 1", '2001-01-07'),
        (1001, 10, "Title 2", "Album 2", "Artist 1", '2001-01-07'),
    ], columns=[
        'discogs_id',
        'release_group_id',
        'title',
        'album',
        'artist',
        'release_date'
    ], index=[1, 2, 3]

    )

    def test_lookup(self):
        spotify = SpotifyListens(self.test_data)
        print(spotify.lookup)
        assert len(spotify.lookup) == 3
        assert ('title 1', 'album 1', 'artist 1') in spotify.lookup

    def test_get_recording_id(self):
        spotify = SpotifyListens(self.test_data)
        spotify_record = {
            'master_metadata_track_name': 'Title 1',
            'master_metadata_album_artist_name': 'Artist 1',
            'master_metadata_album_album_name': 'Album 2',
            'spotify_track_uri': 'spotify:track:1234abcd',  # <- wrong, obvs.
        }
        assert spotify.get_recording_id(spotify_record) == 2

    def test_get_spotify_jazz(self):
        spotify = SpotifyListens(self.test_data)
        spotify_records = [{
            'master_metadata_track_name': 'Title 1',
            'master_metadata_album_artist_name': 'Artist 1',
            'master_metadata_album_album_name': 'Album 2',
            'spotify_track_uri': 'spotify:track:1234abcd',  # <- wrong, obvs.
        },
        {
            'master_metadata_track_name': 'Title 1',
            'master_metadata_album_artist_name': 'Artist 3',
            'master_metadata_album_album_name': 'Album 2',
            'spotify_track_uri': 'spotify:track:1234xyzd',  # <- wrong, obvs.
        }]

        results = list(spotify.get_spotify_jazz(spotify_records))
        record, rec_id = results[0]
        assert record == spotify_records[0]
        assert len(results) == 1
