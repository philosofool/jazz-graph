import pytest
import torch
from torch_geometric.data import HeteroData


@pytest.fixture
def hetero_data() -> HeteroData:
    data = HeteroData()
    data['artist'].x = torch.tensor([3, 1, 0, 2])
    data['song'].x = torch.tensor([10, 9, 11])
    data['performance'].x = torch.tensor([20, 21, 22, 24, 23])

    data['artist', 'performs', 'performance'].edge_index = torch.tensor([
        [1, 1, 3, 3, 3],  # values zero and three missing.
        [0, 1, 0, 1, 2]   # 23 and 24 missing.
    ])
    data['performance', 'performing', 'song'].edge_index = torch.tensor([
        [0, 1, 2, 4], # 24 missing (island); 23 is not an island (but has no performs relation)
        [0, 0, 2, 2]  # song 9 is an island
    ])
    data['artist', 'composed', 'song'].edge_index = torch.tensor([
        [0, 1],  # three is not an island: only composes. zero is an island.
        [0, 2]
    ])

    data['performance'].y = torch.tensor([1, 2, 3, 4, 5]) / 10
    data['artist'].y = torch.tensor([3, 5, 4, 9], dtype=torch.float32)
    return data