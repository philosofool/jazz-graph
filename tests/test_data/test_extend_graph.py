
import torch
from torch_geometric.data import HeteroData
import numpy as np

from jazz_graph.data.graph_transforms import extend_graph

def test_extend_graph():
    graph = HeteroData()
    graph['artist'].x = torch.tensor([1, 2, 3]).reshape(-1, 1)
    graph['performance'].x = torch.tensor([11, 12, 13]).reshape(-1, 1)
    graph['artist', 'performs', 'performance'].edge_index = torch.tensor([
        [0, 0, 1, 2, 2],
        [0, 1, 1, 1, 2]
    ])
    new_nodes = {'artist': {'x': torch.tensor([1]).reshape(-1, 1)}}
    new_edges = {
        ('artist', 'performs', 'performance'):
        {'edge_index': torch.tensor([[3, 3], [0, 2]])}
    }
    result = extend_graph(graph, new_nodes, new_edges)
    assert result is not graph, "Transform should create new graph."
    assert torch.all(result['artist'].x == torch.tensor([1, 2, 3, 1]).reshape(-1, 1))
    assert torch.all(result['performs'].edge_index == torch.tensor([
        [0, 0, 1, 2, 2, 3, 3],
        [0, 1, 1, 1, 2, 0, 2]
    ]))
    with np.testing.assert_raises(ValueError):
        # Adding bad data will cause it to fail.
        result = extend_graph(graph, {}, new_edges)
