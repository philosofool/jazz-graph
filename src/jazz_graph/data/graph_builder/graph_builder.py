import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from jazz_graph.data.graph_transforms import mask_node_degree, prune_graph_from_masks


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
