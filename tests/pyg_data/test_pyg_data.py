import pandas as pd
import numpy as np
import torch
from jazz_graph.pyg_data.pyg_data import torch_values, torch_index, CreateTensors

def test_torch_values():
    df = pd.DataFrame({'a': [1., 2., 3], 'b': [4., 5., 6.]})
    expected = torch.tensor([[1., 4], [2, 5], [3, 6]])
    result = torch_values(df)
    assert isinstance(result, torch.Tensor)
    np.testing.assert_array_equal(result, expected)

    df = pd.DataFrame(index=[0, 1, 2, 3])
    assert torch_values(df).shape == (4, 0), "A dataframe of featureless nodes should have shape (n_nodes, 0)."

def test_torch_index():
    df = pd.DataFrame({'b': [4., 5., 6.]})
    expected = torch.tensor([0, 1, 2]).reshape(-1, 1)
    result = torch_index(df)
    assert isinstance(result, torch.Tensor)
    np.testing.assert_array_equal(result, expected)

    index = pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)])
    df.index = index
    expected = torch.tensor([[1, 2], [3, 4], [5, 6]])
    result = torch_index(df)
    np.testing.assert_array_equal(result, expected)

def _write_parquet(df, filename: str, directory: str = '/workspace'):
    import os
    path = os.path.join(directory, filename)
    df.to_parquet(path, index=True)
    print(f"Wrote {df.shape[0]} rows, {df.shape[1]} columns to {path}.")

class TestCreateTensors:
    path = '/workspace/tests/test_pyg_data'

    songs = pd.DataFrame({'title': [1, 3, 2]})
    artists = pd.DataFrame({'name': [100, 40, 50]})
    performances = pd.DataFrame({'recording_id': [100, 101, 102], 'release_date': pd.to_datetime(['2000', '1956', '1976']), 'free_jazz': [0, 0, 1], 'bop': [1, 0, 1], 'vocal': [0, 1, 1]})
    performance = performances.astype({'release_date': 'object'})  # assure that if source data is object type, casting will work.

    song_artist_edges = pd.DataFrame({'work_id': [0, 1, 2], 'artist_id': [1, 2, 2]})
    performance_artist_edges = pd.DataFrame({'recording_id': [0, 0, 1, 1, 2], 'artist_id': [1, 0, 1, 2, 2]})
    performance_song_edge = pd.DataFrame({'work_id': [0, 1, 2], 'recording_id': [2, 1, 0]})
    import os
    if os.path.exists(path):
        _write_parquet(performances, 'performance_nodes.parquet', path)
        _write_parquet(artists, 'artist_nodes.parquet', path)
        _write_parquet(songs, 'song_nodes.parquet', path)

        _write_parquet(song_artist_edges, 'song_artist_edges.parquet', path)
        _write_parquet(performance_artist_edges, 'performance_artist_edges.parquet', path)
        _write_parquet(performance_song_edge, 'performance_song_edges.parquet', path)

    def test_performances(self):
        create = CreateTensors(self.path)
        assert create.performances().shape == (3, 1)
        assert create.performances().dtype == torch.int32
        np.testing.assert_array_equal(create.performances(), np.array([2000, 1956, 1976]).reshape(-1, 1))

    def test_songs(self):
        create = CreateTensors(self.path)
        expected = torch.tensor([1, 3, 2]).reshape(-1, 1)
        np.testing.assert_array_equal(create.songs(), expected)

    def test_artists(self):
        create = CreateTensors(self.path)
        expected = torch.tensor([100, 40, 50]).reshape(-1, 1)
        np.testing.assert_array_equal(create.artists(), expected)

    def test_performance_artist_edges(self):
        create = CreateTensors(self.path)
        expected = torch.tensor([
            [1, 0, 1, 2, 2],
            [0, 0, 1, 1, 2]
        ])
        np.testing.assert_array_equal(create.artist_performance_edges(), expected)

    def test_performance_song_edges(self):
        create = CreateTensors(self.path)
        expected = np.array([
            [2, 1, 0],
            [0, 1, 2]
        ])
        np.testing.assert_array_equal(create.performance_song_edges(), expected)

    def test_song_artist_edges(self):
        create = CreateTensors(self.path)
        expected = torch.tensor([
            [1, 2, 2],
            [0, 1, 2]
        ])
        np.testing.assert_array_equal(create.artist_song_edges(), expected)

    def test__mask_slices(self):
        create = CreateTensors(self.path)
        create.performances()
        # hacky dependence on implementation but we don't need that much rigor here.
        create._performances = pd.DataFrame(np.arange(0, 40_000).reshape(-1, 4))
        train, dev, test = create._mask_slices()
        assert not np.any(train & dev & test), "The slices should be disjoint."
        assert np.all(train | dev | test), "The slices should be exhaustive."

        # basic size checks, should be approx. ratio 8:1:1.
        assert np.sum(train) > 7500
        assert np.sum(test) < 1200
        assert np.sum(dev) < 1200

    def test_labels(self):
        create = CreateTensors(self.path)
        result = create.labels()
        expected = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]).T
        np.testing.assert_array_equal(result, expected)
