from tempfile import TemporaryFile
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

from torch_geometric.transforms import ToUndirected

from jazz_graph.model.model import JazzModel
from jazz_graph.recommendation.recommender import LookupRecordings, PredictLinkRecommender
import pytest

class TestLookupRecordings:
    # def test_from_path(self):
    #     ...

    def test_lookup_node_index(self):
        data = pd.DataFrame({'ids': [1, 2]}, index=[101, 102])
        lookup = LookupRecordings(data)
        ids = [101, 102]
        result = lookup.lookup_node_index(ids)
        np.testing.assert_array_equal(result, np.array([1, 2]))

        result = lookup.lookup_node_index([101, 103])
        np.testing.assert_array_equal(result, [1])

        with np.testing.assert_raises(KeyError):
            result = lookup.lookup_node_index([101, 103], missing='raise')

    def test_lookup_recording_ids(self):
        data = pd.DataFrame({'ids': [1, 2]}, index=[101, 102])
        lookup = LookupRecordings(data)
        np.testing.assert_array_equal(lookup.lookup_recording_ids(np.array([1])), [102])

@pytest.fixture
def hetero_data() -> HeteroData:
    data = HeteroData()
    data['artist'].x = torch.tensor([103, 101, 100, 102]).reshape(-1, 1)
    data['song'].x = torch.tensor([10, 9, 11]).reshape(-1, 1)
    data['performance'].x = torch.tensor([20, 21, 22, 24, 23]).reshape(-1, 1)

    data['artist', 'performs', 'performance'].edge_index = torch.tensor([
        [1,   1,  3,  3,  3],  # values zero and three missing.
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
    return ToUndirected()(data)

@pytest.fixture
def recommender(hetero_data) -> PredictLinkRecommender:
    num_performances = hetero_data['performance'].num_nodes
    num_artists = hetero_data['artist'].num_nodes
    num_songs = hetero_data['song'].num_nodes
    model = JazzModel(num_performances, num_artists, num_songs, 8, 8, hetero_data.metadata())
    lookup = LookupRecordings(pd.DataFrame(
        {'ids': np.arange(num_performances), 'recording_id': hetero_data['performance'].x.reshape(-1)}
    ).set_index('recording_id'))

    recommender = PredictLinkRecommender(model, hetero_data, lookup)
    return recommender

class TestPredictLinkRecommender:

    def test_get_user_parameters(self, recommender):
        user_listens = [21, 22]
        new_nodes, new_edges, new_embeds = recommender.get_user_parameters(user_listens)
        assert torch.all(new_nodes['artist']['x'] == torch.tensor([[4]]))
        assert torch.all(new_edges[('artist', 'performs', 'performance')]['edge_index'] == torch.tensor([[4, 4], [1, 2]]))
        np.testing.assert_array_equal(new_edges[('performance', 'rev_performs', 'artist')]['edge_index'], torch.tensor([[4, 4], [1, 2]]).flip(0))
        assert new_embeds.shape == (1, 8)

    def test__sort_scores(self, recommender):
        scores = torch.tensor([[.1, -1., -.4, -.33, .72]]).t()
        recs, scores, mask = recommender._sort_scores(scores)
        # expected = [20, 21, 22, 24, 23]
        # expected = [1, 2, 4, 3, 0]
        expected_recs = [23, 20, 24, 22, 21]
        expected_scores = [.72, .1, -.33, -.4, -1.]
        expected_mask = [4, 0, 3, 2, 1]
        np.testing.assert_array_equal(recs, expected_recs)
        np.testing.assert_array_almost_equal(scores, expected_scores)
        np.testing.assert_array_equal(mask, expected_mask)
        assert mask.dtype == np.int64



    def test_inductive_recommend(self, recommender):
        user_listens = [21, 22]
        new_nodes, new_edges, new_embed = recommender.get_user_parameters(user_listens)

        expected_model_weights = recommender.model.artist_embed.weight
        scores, z = recommender.inductive_rec(new_nodes, new_edges, new_embed)
        assert torch.all(expected_model_weights == recommender.model.artist_embed.weight)
