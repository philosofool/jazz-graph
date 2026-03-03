from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

from jazz_graph.data.graph_builder import CreateTensors
from jazz_graph.training.logging import load_embeddings

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


def cosine_similarity(user_embedding: torch.Tensor, performance_embedding: torch.Tensor) -> torch.Tensor:
    """Cosine similarity."""
    with torch.no_grad():
        return (
            performance_embedding @ user_embedding) / (
            torch.linalg.norm(user_embedding) * torch.linalg.norm(performance_embedding, dim=1))

def dot_product_similarity(user_embedding: torch.Tensor, performance_embedding: torch.Tensor) -> torch.Tensor:
    """Cosine similarity."""
    with torch.no_grad():
        return performance_embedding @ user_embedding

def aggregate_user_embeddings(user_embeddings: torch.Tensor) -> torch.Tensor:
    return torch.mean(user_embeddings, dim=0)


class LookupRecordings:
    """Look up recording ids and node indexes.

    Parameters
    ----------
    data:
        A dataframe with a column 'ids' that matches the index
        of the node in the graph and and index matching the
        recording id of the performance.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @classmethod
    def from_path(cls, node_data_path):
        data = cls._get_lookup(node_data_path)
        return cls(data)


    def lookup_node_index(self, listens: list[int], missing='ignore') -> np.ndarray:
        """Get the node index from a recorind id."""
        if missing == 'ignore':
            # Find the subset of listens that are in the index, avoiding key errors.
            listens_subset = self.data.index.intersection(listens)
        else:
            listens_subset = listens
        data = self.data.loc[listens_subset]
        return data['ids'].to_numpy()

    def mask_listens(self, listens: list[int]):
        return self.data.index.isin(listens)

    def lookup_recording_ids(self, indexes: np.ndarray) -> np.ndarray:
        """Get recording ids from a collection of node indexes."""
        return self.data.index[indexes].to_numpy()

    @staticmethod
    def _get_lookup(path):
        create = CreateTensors(path)
        performance_data = create.load_parquet('performance_nodes.parquet')
        ids = np.arange(len(performance_data))
        lookup = pd.DataFrame(ids, index=performance_data.recording_id, columns=['ids'])
        return lookup


class Recommender:
    def __init__(self, embeddings: nn.Embedding, lookup: LookupRecordings):
        self.embeddings = embeddings
        self.lookup_recordings = lookup

    @classmethod
    def from_path(cls, embeddings_path, nodes_data_path):
        lookup_recordings = LookupRecordings.from_path(nodes_data_path)
        embeddings = cls._get_embedding_artifact(embeddings_path)
        return cls(embeddings, lookup_recordings)

    @staticmethod
    def _get_embedding_artifact(str) -> nn.Embedding:
        all_embeds, _ = load_embeddings(str)
        return all_embeds['performance']

    def make_user_embedding(self, listens):
        embedding_indexes = self.lookup_recordings.lookup_node_index(listens)

        relevant_embeddings = self.embeddings(torch.tensor(embedding_indexes))
        user_embedding = aggregate_user_embeddings(relevant_embeddings)
        return user_embedding

    def get_recommendations(self, listens: list[int]):
        user_embedding = self.make_user_embedding(listens)
        similarity_scores = dot_product_similarity(user_embedding, self.embeddings.weight)
        recommendations, scores = self._sort_scores(similarity_scores)
        return recommendations, scores

    def _sort_scores(self, scores) -> tuple[np.ndarray, np.ndarray]:
        scores = scores.view(-1)
        recommendations = torch.argsort(scores, descending=True)
        rec_recordings = self.lookup_recordings.lookup_recording_ids(recommendations.numpy())
        return rec_recordings, scores[recommendations].numpy()


## Inductive Graph Recommender

from jazz_graph.data.graph_transforms import extend_graph
from jazz_graph.data.graph_builder import make_jazz_data, CreateTensors
from jazz_graph.model.model import JazzModel

class PredictLinkRecommender(Recommender):
    def __init__(self, model: JazzModel, data: HeteroData, lookup: LookupRecordings):
        self.model = model
        self.data = data
        self.lookup_recordings = lookup

    def get_user_parameters(self, user_listens: list[int], weight_by_count: bool = False):
        performance_idx = torch.tensor(self.lookup_recordings.lookup_node_index(user_listens))
        num_existing_artists = self.data['artist'].num_nodes
        num_performances = self.data['performance'].num_nodes
        new_edge_index = torch.stack([
            torch.tensor(num_existing_artists).repeat(performance_idx.size(0)),
            performance_idx
        ])
        new_nodes = {'artist': {'x': torch.tensor([[num_existing_artists]])}}
        new_edges = {
            ('artist', 'performs', 'performance'): {'edge_index': new_edge_index},
            ('performance', 'rev_performs', 'artist'): {'edge_index': new_edge_index.flip(0)}
        }
        new_embeds = self._make_artist_embeds(performance_idx, weight_by_count)
        return new_nodes, new_edges, new_embeds

    def _make_artist_embeds(self, indexes, weight_by_count):
        edge_index = self.data['artist', 'performs', 'performance'].edge_index
        mask = torch.isin(edge_index[1], indexes)
        artists = edge_index[0][mask]
        if weight_by_count:
            artist_embeds = self.model.artist_embed(artists)
        else:
            artist_embeds = self.model.artist_embed(artists.unique())
        with torch.no_grad():
            new_embeds = torch.max(artist_embeds, dim=0, keepdim=True)[0]
        return new_embeds

    def inductive_rec(self, new_nodes, new_edges, new_artist_embed):
        orig_num_artists = self.data['artist'].num_nodes
        num_performances = self.data['performance'].num_nodes
        new_artist_indecies = torch.arange(orig_num_artists, orig_num_artists + 1)
        original_performance_indecies = torch.arange(num_performances)

        # candidates: all paths from new artist to existing performances.
        src = new_artist_indecies.repeat_interleave(num_performances)
        dst = original_performance_indecies.repeat(1) # 1 = num_new indecies

        new_data = extend_graph(self.data, new_nodes, new_edge_index=new_edges)
        # add n_id feature to new data?
        for node_type in new_data.metadata()[0]:
            new_data[node_type].n_id = torch.arange(new_data[node_type].num_nodes)


        with AmndedEmbeddings(self.model, new_artist_embed) as model:
            with torch.no_grad():
                z = model(new_data.x_dict, new_data.edge_index_dict, new_data)

        scores = (z['performance'][dst] * z['artist'][src]).sum(-1)
        scores = scores.view(1, num_performances)
        return scores, z

    def get_recommendations(self, listens, weight_by_count: bool = False):
        """Return recommended recording ids and their scores."""
        new_nodes, new_edges, new_embed = self.get_user_parameters(listens, weight_by_count)
        scores, _ = self.inductive_rec(new_nodes, new_edges, new_embed)
        rec_recordings, sorted_scores = self._sort_scores(scores)
        return rec_recordings, sorted_scores



class AmndedEmbeddings:
    def __init__(self, model: JazzModel, new_embed_weights):
        self.new_embed_weights = new_embed_weights
        self.model = model

    def __enter__(self):
        self._old_embed = self.model.artist_embed
        old_weights = self._old_embed.weight
        new_embed = torch.nn.Embedding.from_pretrained(torch.concat([old_weights, self.new_embed_weights]))
        self.model.artist_embed = new_embed
        return self.model

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.artist_embed = self._old_embed
