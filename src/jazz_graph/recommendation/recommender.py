import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import json
import pandas as pd
import numpy as np
from collections.abc import Callable

from jazz_graph.data.graph_builder import CreateTensors


def cosine_similarity(user_embedding: torch.Tensor, performance_embedding: torch.Tensor) -> torch.Tensor:
    """Cosine similarity, at present."""
    with torch.no_grad():
        return (
            performance_embedding @ user_embedding) / (
            torch.linalg.norm(user_embedding) * torch.linalg.norm(performance_embedding, dim=1))

def aggregate_user_embeddings(user_embeddings: torch.Tensor) -> torch.Tensor:
    return torch.mean(user_embeddings, dim=0)

class LookupRecordings:
    def __init__(self, nodes_data_path):
        self.data = self._get_lookup(nodes_data_path)

    def lookup_listen_ids(self, listens: list[int]) -> np.ndarray:
        # TODO: Not fault tollerant--any missing key would cause an error.
        return self.data.loc[listens].ids.to_numpy()

    def lookup_recording_ids(self, ids: np.ndarray) -> np.ndarray:
        return self.data.index[ids].to_numpy()

    @staticmethod
    def _get_lookup(path):
        create = CreateTensors(path)
        performance_data = create.load_parquet('performance_nodes.parquet')
        ids = np.arange(len(performance_data))
        lookup = pd.DataFrame(ids, index=performance_data.recording_id, columns=['ids'])
        return lookup


class Recommender:
    def __init__(self, embeddings_path, nodes_data_path):
        self.lookup_recordings = LookupRecordings(nodes_data_path)
        self.embeddings = self._get_embedding_artifact(embeddings_path)

    def _get_embedding_artifact(self, str) -> nn.Embedding:
        all_embeds, _ = load_embeddings(str)
        return all_embeds['performance']

    def make_user_embedding(self, listens):
        embedding_indexes = self.lookup_recordings.lookup_listen_ids(listens)
        relevant_embeddings = self.embeddings(torch.tensor(embedding_indexes))
        user_embedding = aggregate_user_embeddings(relevant_embeddings)
        return user_embedding


    def get_recommendations(self, listens: list[int]):
        user_embedding = self.make_user_embedding(listens)
        similarity_scores = cosine_similarity(user_embedding, self.embeddings.weight)
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
