from jazz_graph.recommendation.experiment import RandomAlbumSplit, SpotifyExperiment
import numpy as np
# from warnings import ...
from collections import namedtuple
import pandas as pd

# Mock tuples from recording_traits.itertuples
Record = namedtuple('Record', ['Index', 'release_group_id'])

class TestRandomAlbumSplit:

    def test_add_to_splits(self):
        album_split = RandomAlbumSplit(seed=2)
        record = {'recording_id': 1, 'album_id': 10}
        record = Record(1, 10)
        album_split.add_to_splits(record)
        assert 10 in album_split.split_a or 10 in album_split.split_b
        if 10 in album_split.split_a:
            assert 1 in album_split.recordings_a
        else:
            assert 1 in album_split.recordings_b

        record = Record(2, 10)
        album_split.add_to_splits(record)
        if 10 in album_split.split_a:
            assert 2 in album_split.recordings_a
        else:
            assert 2 in album_split.recordings_b

    def test_make_splits(self):
        album_split = RandomAlbumSplit(seed=2)
        data = [Record(x * 1000, x % 100) for x in range(100)]
        album_split.make_splits(data)
        for x in album_split.recordings_a:
            x_root = x // 1000
            assert x_root % 100 in album_split.split_a
        for x in album_split.recordings_b:
            x_root = x // 1000
            assert x_root % 100 in album_split.split_b

        assert album_split.split_a, "There should be some albums in a."
        assert album_split.split_b, "There should be some albums in b."
        assert not album_split.split_a.intersection(album_split.split_b), "They should not overlap."


class TestSpotifyExperiment:
    def test_experiment_metrics(self, recording_traits, spotify_data, tmp_path, monkeypatch):
        tp_novel = np.arange(0, 100)
        fn_novel = np.arange(100, 200)
        exploratory_novel = np.arange(200, 1000)
        tp_familiar = np.arange(1000, 1100)
        fn_familiar = np.arange(1100, 1200)
        exploratory_familiar = np.arange(1200, 2000)

        monkeypatch.setattr(SpotifyExperiment, 'in_samples', property(lambda self: np.arange(1000, 1200)))
        monkeypatch.setattr(SpotifyExperiment, 'out_samples', property(lambda self: np.arange(0, 200)))

        import string
        import random
        def make_gibberish_array(n):
            k = n * 10
            chars = string.ascii_uppercase + string.digits
            gibberish = ''.join(random.choices(chars, k=k))
            gibberish_array = np.array([gibberish[i:i+10] for i in range(0, k, 10)])
            return gibberish_array

        title = make_gibberish_array(2_000)
        artist = make_gibberish_array(2_000)
        albums = make_gibberish_array(2_000)
        recording_traits = pd.DataFrame({'artist': artist, 'title': title, 'album': albums, 'release_date': np.arange(2000)}, index=np.arange(2000))

        spotify_experiment = SpotifyExperiment(recording_traits, {}, tmp_path)

        # Test 1: 0 recall novel, 1.0 recall familiar
        recommendations = np.concat([
            tp_familiar,
            fn_familiar,  # NOTE: not false in this case.
            exploratory_familiar,
            exploratory_novel,
            tp_novel,
            fn_novel,
        ])
        mask = np.isin(recommendations, np.arange(1000, 1200))
        result = spotify_experiment.experiment_metrics(recommendations, mask, 2)
        assert result['recall_familiar'] == 1.
        assert result['recall_novel'] == 0.

        # Test 2: .5 recall familiar and novel
        recommendations = np.concat([
            tp_familiar,
            tp_novel,
            exploratory_familiar,
            exploratory_novel,
            fn_familiar,
            fn_novel,
        ])
        mask = np.isin(recommendations, np.arange(1000, 1200))
        result = spotify_experiment.experiment_metrics(recommendations, mask, 2)
        assert result['recall_familiar'] == .5
        assert result['recall_novel'] == .5

        # Test 3: 1. recall for both

        recommendations = np.concat([
            tp_familiar,
            tp_novel,
            fn_familiar,  # NOTE: not false in this case.
            fn_novel,  # also not false in this case.
            exploratory_familiar,
            exploratory_novel,
        ])
        mask = np.isin(recommendations, np.arange(1000, 1200))
        result = spotify_experiment.experiment_metrics(recommendations, mask, 2)
        assert result['recall_familiar'] == 1.
        assert result['recall_novel'] == 1.

        # Recall is perfect when k is very large.
        # recommendations = np.concat([
        #     tp_familiar,
        #     tp_novel,
        #     exploratory_familiar,
        #     exploratory_novel,
        #     fn_familiar,
        #     fn_novel,
        # ])
        # mask = np.isin(recommendations, np.arange(1000, 1200))
        # with np.testing.assert_warns(UserWarning):
        #     result = spotify_experiment.experiment_metrics(recommendations, mask, 100)
        # assert result['recall_familiar'] == 1.
        # assert result['recall_novel'] == 1.

    def test_coverage(self, recording_traits, spotify_data, tmp_path):
        spotify_experiment = SpotifyExperiment(recording_traits, spotify_data, tmp_path)
        seeded = [
            159729, 159730, 384218,
            270583, 30646764
        ]
        positives = [
            159729, 159730, 384218,
            21881894, 21881892, 270585
        ]
        result = spotify_experiment.coverage(positives, seeded)
        artist_counts = result['artist_counts']
        assert len(artist_counts) == 1, "John Coltrane is the only novel artist."
        assert artist_counts['John Coltrane'] == 1., "All performances are by John Coltrane."
        assert np.isclose(result['novel_frequency'], 1 / 3), "1/3 of all recommendations included and unfamiliar artist."
