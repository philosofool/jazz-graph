from codecs import lookup
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import json
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

from jazz_graph.data.graph_builder import CreateTensors


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
        recommendations = torch.argsort(similarity_scores, descending=True)
        rec_recordings = self.lookup_recordings.lookup_recording_ids(recommendations.numpy())
        return rec_recordings, similarity_scores[recommendations].numpy()


def load_embeddings(embedding_path):
    """Load embeddings for recommendation."""
    embedding_path = Path(embedding_path)
    embeddings = torch.load(embedding_path / "embeddings.pt", weights_only=False)
    with open(embedding_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    return embeddings, metadata
