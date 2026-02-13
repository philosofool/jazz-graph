import os

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import HeteroData

data = HeteroData()
data['performance'].x = ...
data['song'].x = ...
data['artist'].x = ...

data['artist', 'performs', 'performance'].edge_index = ...
data['performance', 'performing', 'song'].edge_index = ...
data['song', 'composes', 'artist'].edge_index = ...

data['artist', 'performs', 'performance'].edge_attr = ...

class CreateTensors:
    def __init__(self, directory):
        self.directory = directory
        self._artists = None
        self._songs = None
        self._performances = None

    def load_parquet(self, path: str) -> pd.DataFrame:
        full_path = os.path.join(self.directory, path)
        df = pd.read_parquet(full_path)
        return df

    def artists(self) -> torch.Tensor:
        if self._artists is None:
            self._artists = self.load_parquet('artist_nodes.parquet')
            # self._artists_lookup = NodeLookUp(self._artists)

        return torch_values(self._artists)

    def songs(self) -> torch.Tensor:
        if self._songs is None:
            self._songs = self.load_parquet('song_nodes.parquet')
            # self._songs_lookup = NodeLookUp(self._songs)
        return torch_values(self._songs)

    def performances(self) -> torch.Tensor:
        if self._performances is None:
            self._performances = self.load_parquet('performance_nodes.parquet')
            # self._performance_lookup = NodeLookUp(self._performances)
        return torch_values(self._performances)

    def song_artist_edges(self) -> torch.Tensor:
        if getattr(self, '_song_artist_edges', None) is None:
            self._song_artist_edges = self.load_parquet('song_artist_edges.parquet')

        return torch_values(self._song_artist_edges).T

    def performance_artist_edges(self) -> torch.Tensor:
        if getattr(self, '_performance_artist_edges', None) is None:
            self._performance_artist_edges = self.load_parquet('performance_artist_edges.parquet')
        return torch_values(self._performance_artist_edges).T

    def performance_song_edges(self) -> torch.Tensor:
        if getattr(self, '_performance_song_edges', None) is None:
            self._performance_song_edges = self.load_parquet('performance_song_edges.parquet')
        return torch_values(self._performance_song_edges).T


def torch_values(df: pd.DataFrame) -> torch.Tensor:
    """Convert values in the dataframe to a single tensor."""
    return torch.tensor(df.to_numpy())

def torch_index(df: pd.DataFrame) -> torch.Tensor:
    """Convert the values in the dataframe index to a single tensor."""
    if isinstance(df.index, pd.MultiIndex):
        data = df.index.to_list()
        data = np.array(data, dtype=np.int64)
    else:
        data = df.index.to_numpy()
        assert len(data.shape) == 1
        data = data.reshape(-1, 1)
    return torch.tensor(data)


if __name__ == '__main__':

    create = CreateTensors('/workspace/local_data/')
    create.songs()
    create.artists()
    create.song_artist_edges().shape

    torch.min(create.performance_artist_edges())
    torch.max(create.performance_artist_edges())


    data = HeteroData()
    data['performance'].x = ...
    data['song'].x = ...
    data['artist'].x = ...

    data['artist', 'performs', 'performance'].edge_index = ...
    data['performance', 'performing', 'song'].edge_index = ...
    data['song', 'composes', 'artist'].edge_index = ...

    data['artist', 'performs', 'performance'].edge_attr = ...