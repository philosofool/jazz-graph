
from collections.abc import Hashable
import torch
from torch_geometric.data import HeteroData


def extend_graph(data: HeteroData, new_nodes: dict[str, dict[Hashable, torch.Tensor]], new_edge_index: dict[str, dict[Hashable, torch.Tensor]]):
    data = data.clone()
    for node_type, feature_dict in new_nodes.items():
        node_data = data[node_type]
        for feature, new_data in feature_dict.items():
            node_data[feature] = torch.concat([node_data[feature], new_data])

    for edge_type, feature_dict in new_edge_index.items():
        assert edge_type in data.metadata()[1]
        edge_data = data[edge_type]
        for feature, new_data in feature_dict.items():
            edge_data[feature] = torch.concat([edge_data[feature], new_data], dim=1)
    data.validate()
    return data


def drop_edge_from_masks(src_graph, edge_masks, dst_graph):
    # This is a weird function. I see the value in augmenting data,
    # but it's somewhat awkward--specify the result from a source
    # but don't you could easily create an invalid graph from this.
    # Maybe just move to unsupervised...
    for edge_type, mask in edge_masks.items():
        src_edge_index = src_graph[edge_type].edge_index
        dst_graph[edge_type].edge_index = src_edge_index[:, mask]
    return None


def map_to_new_node_index(edge_index, node_mask: torch.Tensor) -> torch.Tensor:
    """Remap the values in edge index to point to values in a new tensor
    that contains only values the nodes in nodes mask.
    """
    new_node_index = node_mask.to(torch.int64)
    new_node_index[0] = new_node_index[0] - 1
    new_node_index = torch.cumsum(new_node_index, dim=0)
    edges_to_keep = node_mask[edge_index]
    new_edge = new_node_index[edge_index]
    return new_edge[edges_to_keep]


def prune_graph_from_masks(src_graph: HeteroData, masks: dict):
    """Prune the graph to contain only selected nodes corresponding edges, returning new data."""
    dst_graph = HeteroData()
    node_types, edge_types = src_graph.metadata()
    for node_type in node_types:
        node_data = src_graph[node_type]
        for key, tensor in node_data.items():
            mask = masks[node_type]
            tensor = tensor[mask]
            dst_graph[node_type][key] = tensor
    # set the edge indexes in out.
    for src, relation, dst in edge_types:
        edge = src_graph[src, relation, dst]
        # currently assumes edge_index is only edge property.
        src_indexes = edge.edge_index[0]
        dst_indexes = edge.edge_index[1]
        new_edge_index = torch.stack([
            map_to_new_node_index(src_indexes, masks[src]),
            map_to_new_node_index(dst_indexes, masks[dst])
        ])
        dst_graph[src, relation, dst].edge_index = new_edge_index
    return dst_graph
