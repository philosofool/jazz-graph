import pandas as pd
import numpy as np
import torch
from jazz_graph.data.graph_builder.graph_builder import prune_isolated_nodes, torch_values, torch_index

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

def test_prune_isolated_nodes(hetero_data):
    data = hetero_data
    result = prune_isolated_nodes(data)

    assert torch.all(result['artist'].x == torch.tensor([3, 1, 2]))
    assert torch.all(result['song'].x == torch.tensor([10, 11]))
    assert torch.all(result['performance'].x == torch.tensor([20, 21, 22, 23]))
    assert torch.all(result['performance'].y == torch.tensor([1, 2, 3, 5]) / 10), "All features associated with performance should be updated."
    assert torch.all(result['artist'].y == torch.tensor([3, 5, 9], dtype=torch.float32)), "Features in artist labels should be dropped where the node is an island."

    expected_edges = hetero_data.metadata()[1]
    assert expected_edges == result.metadata()[1], "The result should have the same edges in this case."

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
