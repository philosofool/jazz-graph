# Definte augmentation Strategies


import torch
from jazz_graph.data.graph_transforms import drop_edge_from_masks, prune_graph_from_masks

from torch_geometric.data import HeteroData
from torch_cluster import random_walk


def performance_album_map(batch: HeteroData):
    """Return the index of the input batch after a random walk."""
    performances = torch.arange(batch['performance'].num_nodes)
    recordings = batch['performance'].album_id + performances.size(0)
    row = torch.concat([performances, recordings])
    col = torch.concat([recordings, performances])
    mapped_nodes = random_walk(row, col, performances, 2)
    assert isinstance(mapped_nodes, torch.Tensor)
    return mapped_nodes[:, 2]


class MatchAlbumAugmentation:
    def __init__(self, recording_album_edge: torch.Tensor):
        self.recording_album_edge = recording_album_edge

    def map_nodes(self, x_dict: HeteroData) -> torch.Tensor:
        perf = x_dict['performance']
        n_ids = perf.n_id
        recording_album_edge = self._mask_edges(n_ids)
        mapped_nodes = random_walk(recording_album_edge[0], recording_album_edge[1], n_ids, 2)
        assert isinstance(mapped_nodes, torch.Tensor)
        return mapped_nodes[:, 1]

    def _mask_edges(self, n_ids) -> torch.Tensor:
        edge_mask = torch.isin(self.recording_album_edge[0], n_ids)
        return self.recording_album_edge[:, edge_mask]


    def new_view(self, graph: HeteroData) -> HeteroData:
        raise NotImplementedError()
        out = graph.clone()
        map_node_idx = self.map_nodes(graph)
        for src, relation, dst in graph.edge_types:
            if src == 'performance':
                row_idx = 0
            elif dst == 'performance':
                row_idx = 1
            else:
                continue
            edge_map = out[src, relation, dst].edge_index
            row_edge = edge_map[row_idx]
            new_edge = map_node_idx[row_edge]
            edge_map[row_idx] = new_edge
            out[src, relation, dst].edge_index = edge_map.clone()
        for feature, tensor in graph['performance'].items():
            tensor = tensor[map_node_idx[:tensor.size(0)]]
            out['performance'][feature] = tensor.clone()
        return out


def drop_edge_augmentation(graph: HeteroData, dst_graph, drop_edge_prob: float = .2):
    edge_types = graph.metadata()[1]
    edge_masks = {
        edge_type: torch.rand(graph[edge_type].edge_index.size(1)) > drop_edge_prob
        for edge_type in edge_types
    }
    drop_edge_from_masks(graph, edge_masks, dst_graph)


def drop_random_nodes_and_edges(data: HeteroData, drop_edge_prob: float = .5):
    out = data.clone()
    drop_edge_augmentation(data, out, drop_edge_prob=drop_edge_prob)
    # drop_node_augmentation(data, out)  # This would be complex: graphs need to align their node indecies in loss.
    return out


def drop_node_augmentation(src_graph: HeteroData, dst_graph: HeteroData, drop_node_prob: float = .1):
    node_types, edge_types = src_graph.metadata()
    masks = {
        node_type: torch.rand(src_graph[node_type].num_nodes) > drop_node_prob
        for node_type in node_types
    }
    prune_graph_from_masks(src_graph, masks)