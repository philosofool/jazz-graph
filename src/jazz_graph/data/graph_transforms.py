
import torch
from collections.abc import Hashable

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
