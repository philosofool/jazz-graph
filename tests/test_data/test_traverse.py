from jazz_graph.data.traverse import bfs_hetero_with_depth
from torch_geometric.data import HeteroData
import torch

def test_bfs_hetero_with_depth():
    data = HeteroData()
    data['a'].x = torch.tensor([100, 200, 300])
    data['b'].x = torch.tensor([101, 201, 301])
    data['c'].x = torch.tensor([400, 401, 402])
    data['a', 'relates', 'b'].edge_index = torch.tensor([
        [0, 0, 1, 1],
        [1, 2, 2, 0]
    ])
    data['b', 'likes', 'c'].edge_index = torch.tensor([
        [0, 1],
        [0, 0]
    ])
    result = list(bfs_hetero_with_depth(data))
    assert result[0] == (0, 'a', 0), "Start node should be the first a node, depth 0"
    assert result[1] == (1, 'b', 1)
    assert result[2] == (2, 'b', 1)
    assert result[3] == (0, 'c', 2)
    assert len(result) == 4

    result = list(bfs_hetero_with_depth(data, 1, 'a'))
    assert result == [(1, 'a', 0), (2, 'b', 1), (0, 'b', 1), (0, 'c', 2)]

    data['c', 'likes', 'a'].edge_index = torch.tensor([
        [0],
        [1]
    ])
    result = list(bfs_hetero_with_depth(data))
    # NOTE: (2, 'b') would be seen again, but it's alread in the list.
    #       This test checks that.
    assert result[4] == (1, 'a', 3)
    assert result[5] == (0, 'b', 4)
    assert (2, 'b', 4) not in result, "This would imply adding an already seen node."
    assert len(result) == 6
