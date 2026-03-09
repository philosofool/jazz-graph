
import torch
from torch_geometric.data import HeteroData
import numpy as np

from jazz_graph.data.graph_transforms import mask_node_degree, prune_graph_from_masks
from jazz_graph.data.graph_transforms import drop_edge_from_masks, extend_graph, map_to_new_node_index

import pytest

@pytest.fixture
def graph():
    graph = HeteroData()
    graph['artist'].x = torch.tensor([1, 2, 3]).reshape(-1, 1)
    graph['performance'].x = torch.tensor([11, 12, 13]).reshape(-1, 1)
    graph['artist', 'performs', 'performance'].edge_index = torch.tensor([
        [0, 0, 1, 2, 2],
        [0, 1, 1, 1, 2]
    ])
    return graph


def test_extend_graph(graph):
    new_nodes = {'artist': {'x': torch.tensor([1]).reshape(-1, 1)}}
    new_edges = {
        ('artist', 'performs', 'performance'):
        {'edge_index': torch.tensor([[3, 3], [0, 2]])}
    }
    result = extend_graph(graph, new_nodes, new_edges)    # pyright: ignore
    assert result is not graph, "Transform should create new graph."
    assert torch.all(result['artist'].x == torch.tensor([1, 2, 3, 1]).reshape(-1, 1))
    assert torch.all(result['performs'].edge_index == torch.tensor([
        [0, 0, 1, 2, 2, 3, 3],
        [0, 1, 1, 1, 2, 0, 2]
    ]))
    with np.testing.assert_raises(ValueError):
        # Adding bad data will cause it to fail.
        result = extend_graph(graph, {}, new_edges)  # pyright: ignore

def test_drop_edge_from_mask(graph):
    src_graph = graph
    new_edge = torch.tensor([[0], [0]])
    rev_perf = ('performance', 'rev_performs', 'arstist')
    src_graph[rev_perf].edge_index = new_edge
    masks = {
        ('artist', 'performs', 'performance'): torch.tensor([True, False, True, True, False]),
        rev_perf: torch.tensor([True])
    }
    dst_graph = HeteroData()
    result = drop_edge_from_masks(src_graph, masks, dst_graph)
    assert result == None, "The function should mutate dst_graph and return None."

    np.testing.assert_array_equal(dst_graph['performs'].edge_index, torch.tensor([
        [0, 1, 2],
        [0, 1, 1]
    ]))
    assert torch.all(dst_graph[rev_perf].edge_index == new_edge)


def test_map_to_new_node_index():
    old_edge = torch.tensor([0, 1, 2, 0, 2, 1, 2, 3])
    nodes_mask = torch.tensor([True, False, True, False])
    expected = torch.tensor([0, 1, 0, 1, 1])
    result = map_to_new_node_index(old_edge, nodes_mask)
    np.testing.assert_array_equal(result, expected)

    old_edge = torch.tensor([0, 1, 2, 0, 2, 1, 2])
    nodes_mask = torch.tensor([False, False, False])
    expected = torch.tensor([])
    result = map_to_new_node_index(old_edge, nodes_mask)
    np.testing.assert_array_equal(result, expected)

    old_edge = torch.tensor([0, 1, 2, 0, 2, 1, 2])
    nodes_mask = torch.tensor([False, False])
    with np.testing.assert_raises(IndexError):
        # Assure raises an error if the nodes mask is shorter than nodes.
        map_to_new_node_index(old_edge, nodes_mask)


def test_mask_node_degree(hetero_data):
    result = mask_node_degree(hetero_data)
    assert result['artist'].dtype == torch.bool
    assert torch.all(result['artist'] == torch.tensor([1, 1, 0, 1]))
    assert torch.all(result['performance'] == torch.tensor([1, 1, 1, 0, 1]))
    assert torch.all(result['song'] ==  torch.tensor([1, 0, 1]))

    result = mask_node_degree(hetero_data, min_degree=2)
    assert torch.all(result['artist'] ==  torch.tensor([0, 1, 0, 1])), "Artist 3 is an island, artist 0 has one composition edge."


def test_prune_graph_from_masks(hetero_data):
    # NOTE: this is currently the same as test_graph_builder.test_prune_isolated_nodes
    # except that we contruct the mask directly.
    mask = {
        'artist': torch.tensor([1, 1, 0, 1]),
        'performance': torch.tensor([1, 1, 1, 0, 1]),
        'song': torch.tensor([1, 0, 1])
    }
    result = prune_graph_from_masks(hetero_data, mask)
    assert torch.all(result['artist'].x == torch.tensor([3, 1, 2]))
    assert torch.all(result['song'].x == torch.tensor([10, 11]))
    assert torch.all(result['performance'].x == torch.tensor([20, 21, 22, 23]))
    assert torch.all(result['performance'].y == torch.tensor([1, 2, 3, 5]) / 10), "All features associated with performance should be updated."
    assert torch.all(result['artist'].y == torch.tensor([3, 5, 9], dtype=torch.float32)), "Features in artist labels should be dropped where the node is an island."

    expected_performs = torch.tensor([
        [1,   1,  2,  2,  2],  # artist formerly at 3 is left shifted one.
        [0, 1, 0, 1, 2]   # These are all he same.
    ])
    expected_performing = torch.tensor([
        [0, 1, 2, 3], # performance formerly at 4 is left shifted.
        [0, 0, 1, 1]  # song formerly at 2 is left shited.
    ])
    expected_composed = torch.tensor([
        [0, 1],  # No shift from artists.
        [0, 1]  # song formerly at 2 is left shifted.
    ])
    np.testing.assert_array_equal(result['performs'].edge_index, expected_performs)
    np.testing.assert_array_equal(result['performing'].edge_index, expected_performing)
    np.testing.assert_array_equal(result['composed'].edge_index, expected_composed)