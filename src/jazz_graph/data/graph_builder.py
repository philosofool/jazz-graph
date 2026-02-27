import os

from networkx import min_edge_cover
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree


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

        return torch_values(self._artists)

    def songs(self) -> torch.Tensor:
        if self._songs is None:
            self._songs = self.load_parquet('song_nodes.parquet')
        return torch_values(self._songs)

    def labels(self) -> torch.Tensor:
        if getattr(self, '_labels', None) is None:
            self._labels = self.load_parquet('performance_nodes.parquet')
            # NOTE: This is pretty fragile and not currently under a robust test.
            # We should prbably get the expceted label schema under control somewhere.
            cols = [column for column in self._labels if column not in  {'release_date', 'recording_id'}]
            self._labels = self._labels[cols].copy()
        return torch_values(self._labels)

    def label_names(self) -> list:
        if not hasattr(self, '_labels'):
            self.labels()
        return self._labels.columns.to_list()

    def _mask_slices(self, seed=42) -> tuple:
        # NOTE: this is not probably best for the final version.
        # I'm simiplifying for the prototype.
        # we should probably mask whole albums. The large majority
        # of performances on an album have identical personal and
        # differ only in song BUT style info is also usually
        # done at the album level. The task of style classification
        # or edge prediction is probably too easy if that's all it
        # takes--you would rarely need more than the immediate
        # neighborhood of a node to do the prediction. So:
        # FIXME: these masks should be done at the album grouping level.
        size = self.performances().size(0)
        if size > 15_000:
            raise NotImplementedError("This data is not production ready and should only be used with prototyping date right now.")
        rng = np.random.default_rng(seed)
        train_idx = rng.random(size) < .8
        test_idx = ~train_idx & (rng.random(size) < .5)
        dev_idx = ~train_idx & ~test_idx
        return train_idx, dev_idx, test_idx

    def train_mask(self) -> torch.Tensor:
        return self._mask_slices()[0]

    def dev_mask(self) -> torch.Tensor:
        return self._mask_slices()[1]

    def test_mask(self) -> torch.Tensor:
        return self._mask_slices()[2]


    def performances(self) -> torch.Tensor:
        if self._performances is None:
            self._performances = self.load_parquet('performance_nodes.parquet')
            self._performances['release_date'] = self._performances.release_date.astype('datetime64[ms]').dt.year
            self._performances = self._performances[['release_date']].copy()
        return torch_values(self._performances)

    def artist_song_edges(self) -> torch.Tensor:
        if getattr(self, '_song_artist_edges', None) is None:
            df = self.load_parquet('song_artist_edges.parquet')
            self._artist_song_edges = df[['artist_id', 'work_id']].copy()

        return torch_values(self._artist_song_edges).T

    def artist_performance_edges(self) -> torch.Tensor:
        if getattr(self, '_performance_artist_edges', None) is None:
            df = self.load_parquet('performance_artist_edges.parquet')
            self._artist_performance_edges = df[['artist_id', 'recording_id']].copy()
        return torch_values(self._artist_performance_edges).T

    def performance_song_edges(self) -> torch.Tensor:
        if getattr(self, '_performance_song_edges', None) is None:
            df = self.load_parquet('performance_song_edges.parquet')
            self._performance_song_edges = df[['recording_id', 'work_id']].copy()
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

def prune_island_nodes(data: HeteroData):
    """Remove all nodes with degree 0."""
    node_types, edge_types = data.metadata()
    masks = mask_node_degree(data, min_degree=1)
    out = HeteroData()
    for node_type in node_types:
        node_data = data[node_type]
        for key, value in node_data.items():
            mask = masks[node_type]
            value = value[mask]
            out[node_type][key] = value
    for edge_type in edge_types:
        edge = data[edge_type]
        for k, v in edge.items():
            out[edge_type][k] = v
    return out

def mask_node_degree(data: HeteroData, min_degree=1):
    node_types, edge_types = data.metadata()
    seen_relations = set()
    total_degrees = {node_type: torch.zeros(data[node_type].num_nodes, dtype=torch.bool) for node_type in node_types}
    for edge_type in edge_types:
        src, _, dst = edge_type
        edge = data[edge_type].edge_index
        n_src_nodes = data[src].num_nodes
        total_degrees[src] = total_degrees[src] + degree(edge[0], num_nodes=n_src_nodes)
        n_dst_nodes = data[dst].num_nodes
        total_degrees[dst] = total_degrees[dst] + degree(edge[1], num_nodes=n_dst_nodes)
    return {k: v >= min_degree for k, v in total_degrees.items()}
