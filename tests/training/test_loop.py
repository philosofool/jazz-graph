from jazz_graph.training.loop import NeighborLoaderWithJitter, JitterInputs, torch as _torch
import torch
from torch_geometric.data import HeteroData

class TestJitterInputs:
    def test_init(self):
        years = torch.tensor([1956, 1961, 1960])
        jitterer = JitterInputs(years, jitter=3.0)
        assert jitterer.inputs.dtype == torch.float
        assert len(jitterer) == 3

    def test_iter(self):
        years = torch.tensor([1956, 1961, 1960])
        jitterer = JitterInputs(years, jitter=3.0)
        for idx in jitterer:
            assert idx.dtype == torch.int64

    def test_set_epoch(self, monkeypatch):
        def mock_rand(_):
            return torch.tensor([.1, .1, .4])

        monkeypatch.setattr(_torch, 'rand', mock_rand)

        years = torch.tensor([1956, 1961, 1960])
        jitterer = JitterInputs(years, jitter=3.0)
        jitterer.set_epoch(1)
        expected = [0, 2, 1]
        assert list(iter(jitterer)) == expected, "With jitter 3.0, the dates should be sorted by their values; jitter is too small to shift."

        jitterer = JitterInputs(years, jitter=4.0)
        jitterer.set_epoch(1)
        expected = [0, 1, 2]
        assert list(iter(jitterer)) == expected, "With jitter 4.0, this effect is larger affect the order."

class TestNeighborLoaderWithJitter:
    def test_init_handles_tuple_type(self, hetero_data):
        years = hetero_data['performance'].x
        loader = NeighborLoaderWithJitter(hetero_data, ('performance', years), [3, 3], 1)
        batch = next(iter(loader))
        assert isinstance(batch, HeteroData)
