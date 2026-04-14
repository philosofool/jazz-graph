import os

from networkx import min_edge_cover
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from jazz_graph.data.graph_transforms import mask_node_degree, prune_graph_from_masks

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

    def performances(self) -> torch.Tensor:
        if self._performances is None:
            self._performances = self.load_parquet('performance_nodes.parquet')
            self._performances['release_date'] = self._performances.release_date.astype('datetime64[ms]').dt.year
            self._performances = self._performances[['release_date', 'recording_id', 'release_group_id']].copy()
        return torch_values(self._performances[['release_date', 'recording_id',]])

    def album_ids(self) -> torch.Tensor:
        if self._performances is None:
            self.performances()
        assert self._performances is not None, "performances method is expected to create the _performances data."
        return torch.tensor(self._performances.release_group_id.values)

    def labels(self) -> torch.Tensor:
        if getattr(self, '_labels', None) is None:
            self._labels = self.load_parquet('performance_nodes.parquet')
            # NOTE: This is pretty fragile and not currently under a robust test.
            # We should prbably get the expceted label schema under control somewhere.
            cols = [column for column in self._labels if column not in  {'release_date', 'recording_id', 'release_group_id'}]
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
        # of performances on an album have identical personel and
        # differ only in song BUT style info is also usually
        # done at the album level. The task of style classification
        # or edge prediction is probably too easy if that's all it
        # takes--you would rarely need more than the immediate
        # neighborhood of a node to do the prediction. So:
        # FIXME: these masks should be done at the album grouping level.
        # NOTE: Not sure about the above note. The task seems difficult
        # enough that masking whole albums may be unnecessary.
        size = self.performances().size(0)
        # if size > 15_000:
        #     raise NotImplementedError("This data is not production ready and should only be used with prototyping date right now.")
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

def prune_isolated_nodes(data: HeteroData):
    """Remove all nodes with degree 0."""
    masks = mask_node_degree(data, min_degree=1)
    out = prune_graph_from_masks(data, masks)
    return out
    # set the node data in out.

def make_jazz_data(create: CreateTensors) -> HeteroData:
    """Build graph data with limited node features.

    This function was used extensively for training GNNs that worked primarily
    on the topological features of the network. Classifier Models,
    LinkPrediction models, and self-supervised GNNs without edge features.
    """
    data = HeteroData()

    # This is a little clunky. The nodes are not expected to provide
    # substantial feature information--the information is the graph.
    data['performance'].x = create.performances()
    data['song'].x = create.songs()
    data['artist'].x = create.artists()

    data['artist', 'composed', 'song'].edge_index = create.artist_song_edges()
    data['artist', 'performs', 'performance'].edge_index = create.artist_performance_edges()
    data['performance', 'performing', 'song'].edge_index = create.performance_song_edges()

    data['performance'].y = create.labels()
    data['performance'].album_id = create.album_ids()
    data['performance'].train_mask = create.train_mask()
    data['performance'].dev_mask = create.dev_mask()
    data['performance'].test_mask = create.test_mask()

    # TODO: maybe? add instrument attributes on edges.
    # data['artist', 'performs', 'performance'].edge_attr = <instrument>
    data = prune_isolated_nodes(data)
    data = ToUndirected()(data)
    data.validate()
    return data

def make_inter_node_edges(data: pd.DataFrame, link_on: str) -> np.ndarray:
    links = {}
    row_idx = 0
    out = []
    for idx, row in data.iterrows():
        link_value = row[link_on]
        known_links = links.get(link_value)
        if known_links is None:
            links[link_value] = [row_idx]
        else:
            for known_link in known_links:
                out.append([known_link, row_idx])
            links[link_value].append(row_idx)
        row_idx += 1
    return np.array(out).T
