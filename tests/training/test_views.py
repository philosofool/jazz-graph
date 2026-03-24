from jazz_graph.training.views import MatchAlbumAugmentation
import torch
import numpy as np

import pytest

class TestMatchAlbumAugmentation:
    def test_map_nodes(self, hetero_data, monkeypatch):
        recording_to_album = torch.tensor([
            [0, 1, 2, 3, 4, 5], # Note: index five is missing from batch.
            [1, 0, 1, 1, 0, 1]
        ])
        augment = MatchAlbumAugmentation(recording_to_album)
        hetero_data['performance'].n_id = torch.arange(hetero_data['performance'].num_nodes)
        from jazz_graph.training import views
        def fake_random_walk(row, col, nodes, walks):
            nodes_ = nodes.clone()
            nodes[0] = nodes[2]
            nodes[2] = 0
            return torch.stack([nodes_, nodes], dim=1)
        monkeypatch.setattr(views, 'random_walk', fake_random_walk)
        result = augment.map_nodes(hetero_data)
        np.testing.assert_array_equal(result, torch.tensor([2, 1, 0, 3, 4]))

    def test__mask_n_ids(self):
        recording_to_album = torch.tensor([
            [0, 1, 2, 3, 4, 5], # Note: index five is missing from batch.
            [1, 0, 1, 1, 0, 1]
        ])
        augment = MatchAlbumAugmentation(recording_to_album)
        result = augment._mask_edges(torch.tensor([0, 1]))
        np.testing.assert_array_equal(result, torch.tensor([[0, 1], [1, 0]]))
        # The following are unexpected in the current implementation, but test nevertheless.
        result = augment._mask_edges(torch.tensor([5, 5, 0]))
        np.testing.assert_array_equal(result, torch.tensor([[0, 5], [1, 1]])), "Node indecies should not need to be ordered or non-redundant."
        result = augment._mask_edges(torch.tensor([6]))
        assert result.shape == (2, 0), "Missing indecies would result in an empty array (i.e., no value for the missing ones.)"

    def test_new_view(self, hetero_data, monkeypatch):
        recording_to_album = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 0, 1, 1, 0]
        ])
        augment = MatchAlbumAugmentation(recording_to_album)
        with np.testing.assert_raises(NotImplementedError):
            augment.new_view(hetero_data)
        return
        from jazz_graph.training import views
        def fake_random_walk(row, col, nodes, walks):
            nodes_ = nodes.clone()
            nodes[0] = nodes[2]
            nodes[2] = 0
            return torch.stack([nodes_, nodes], dim=1)
        monkeypatch.setattr(views, 'random_walk', fake_random_walk)
        result = augment.new_view(hetero_data)

        expected_performs = torch.tensor([
            [1, 1, 3, 3, 3],  # values zero and three missing.
            [2, 1, 2, 1, 0]   # 23 and 24 missing.
        ])
        expected_performing = torch.tensor([
            [2, 1, 0, 4], # 24 missing (island); 23 is not an island (but has no performs relation)
            [0, 0, 2, 2]  # song 9 is an island
        ])
        expected_composed = torch.tensor([
            [0, 1],
            [0, 2]
        ])

        expected_y = torch.tensor([3, 2, 1, 4, 5]) / 10
        np.testing.assert_array_equal(result['performs'].edge_index, expected_performs)
        np.testing.assert_array_equal(result['performing'].edge_index, expected_performing)
        np.testing.assert_array_equal(result['composed'].edge_index, expected_composed)
        np.testing.assert_array_equal(result['performance'].y, expected_y)
        np.testing.assert_array_equal(result['performance'].x, torch.tensor([22, 21, 20, 24, 23]))
