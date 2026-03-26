from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias
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


Recommendations: TypeAlias = tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.int64]]]

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

    def mask_data_listens(self, listens: list[int] | np.ndarray) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        return self.data.index.isin(listens)

    def mask_node_listens(self, listens: list[int] | np.ndarray) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        mask = self.mask_data_listens(listens)
        return self.data['ids']

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

    def get_recommendations(self, listens: list[int]) -> Recommendations:
        user_embedding = self.make_user_embedding(listens)
        similarity_scores = dot_product_similarity(user_embedding, self.embeddings.weight)
        recommendations, scores, mask = self._sort_scores(similarity_scores)
        return recommendations, scores, mask

    def _sort_scores(self, scores) -> Recommendations:
        scores = scores.view(-1)
        # return the node indexes, sorted by score.
        recommendations = torch.argsort(scores, descending=True)
        rec_recordings = self.lookup_recordings.lookup_recording_ids(recommendations.numpy())
        return rec_recordings, scores[recommendations].numpy(), recommendations.numpy()

class InferenceRecommender(Recommender):
    """Get recommendations from a JazzModel.

    The JazzModel should return embeddings where the dot product of two embeddings
    represents the similarity of the correspondings songs.

    This is used for testing and prototyping, where we want to use an in-memory
    model, rather than cached embeddings, to make an inference.
    """
    def __init__(self, model: JazzModel, data: HeteroData, lookup: LookupRecordings):
        self.model = model
        self.data = data
        self.lookup_recordings = lookup
        for node_type in data.metadata()[0]:
            node = self.data[node_type]
            node.n_id = torch.arange(node.x.size(0))

    @torch.no_grad()
    def get_recommendations(self, listens: list[int]) -> Recommendations:
        """Get recommendations based on input recording ids.

        Returns
        -------
        A tuple of numpy arrays, recording_id, score and a boolean mask.
        The response recording_ids are included. The mask is True where an
        entry was included in the inputs.
        """
        self.model.eval()
        x_dict, edge_index_dict = self.data.x_dict, self.data.edge_index_dict
        performance_embed = self.model(x_dict, edge_index_dict, self.data)['performance']

        familiar_nodes = self.lookup_recordings.lookup_node_index(listens)
        familiar_perf = performance_embed[familiar_nodes]
        # novel_perf = performance_embed
        raw_scores = (performance_embed @ familiar_perf.T)
        scores = raw_scores.sum(dim=-1)

        listens_mask = self.lookup_recordings.mask_data_listens(listens)
        rec_recordings, scores_sorted, sort_index = self._sort_scores(scores)

        return rec_recordings, scores_sorted, listens_mask[sort_index]



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
            new_embeds = artist_embeds.mean(dim=0, keepdim=True)
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

        # This is obviously a hack. It's safe, but there should be a better way...
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
        rec_recordings, sorted_scores, _ = self._sort_scores(scores)
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


class RandomWalkRecommender(Recommender):
    """Recommend performances by a two-hop random walk of the graph.

    The walk hops via any relation edge to an adjacent node then hops back to
    any node of the same type as the origin node.

    Used as a baseline model for making recommendations.
    """
    def __init__(self, data: HeteroData, lookup: LookupRecordings, seed: int|None):
        self.data = data
        self.node_type = 'performance'
        self.lookup_recordings = lookup
        self._seed = seed

    def get_recommendations(self, listens):
        """Get recommended recording_ids, their scores and a mask of input ids."""
        node_ids = self.lookup_recordings.lookup_node_index(listens)
        random_walks = self.heterogeneous_two_hop_random_walk(torch.from_numpy(node_ids), seed=self._seed)
        rec_node_ids = random_walks[:, 1]
        recording_ids = self.lookup_recordings.lookup_recording_ids(rec_node_ids.numpy().tolist())
        recording_ids, mask = self._dedup_and_mask_listens(recording_ids, listens)
        return recording_ids, np.zeros_like(recording_ids), mask

    def _dedup_and_mask_listens(self, recording_ids, listens):
        _, rec_idx = np.unique(recording_ids, return_index=True)
        recording_ids = recording_ids[np.sort(rec_idx)]
        mask = np.isin(recording_ids, listens)
        return recording_ids, mask

    # @staticmethod
    def heterogeneous_two_hop_random_walk(
        self,
        # data: HeteroData,
        # node_type: str,
        node_indices: torch.Tensor,
        num_walks: int = 10,
        return_intermediate: bool = False,
        seed: int | None = None
    ) -> torch.Tensor:
        # Claude.ai provided this code.
        """
        Perform two-hop random walks on a HeteroData graph, starting and ending
        at nodes of the same type (origin -> intermediate -> origin-type node).

        Each walk follows one outgoing (or incoming) edge to an intermediate node
        of a different type, then follows one edge back to a node of the original
        type. The destination may be the origin node itself.

        Args:
            data:               A PyTorch Geometric HeteroData object.
            node_type:          The starting (and ending) node type, e.g. 'artist'.
            node_indices:       1-D tensor of node indices to start walks from.
            num_walks:          Number of walks to attempt per source node.
            return_intermediate: If True, also return the intermediate node tensor.

        Returns:
            walks: LongTensor of shape (N * num_walks, 2) where columns are
                [source_node_index, destination_node_index], both in the
                coordinate space of `node_type`. Invalid/dead-end walks are
                represented as -1.
            (optional) intermediates: LongTensor of shape (N * num_walks,) with
                the intermediate node index visited, -1 for dead ends.
        """
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        node_type = self.node_type
        data = self.data
        node_indices = node_indices.long()
        N = node_indices.size(0)
        total = N * num_walks

        # Expand source indices so each node appears num_walks times.
        sources = node_indices.repeat_interleave(num_walks)  # (total,)

        # Collect all edge relations that touch `node_type`.
        # We separate them into:
        #   forward: node_type -[rel]-> other_type
        #   backward: other_type -[rel]-> node_type  (we traverse in reverse)
        forward_edges = []  # (src_indices, dst_indices, other_type)
        backward_edges = []  # (other_indices, dst_indices, other_type) stored as
        #                      other->node_type adjacency for the return hop

        for (src_type, rel, dst_type), edge_index in data.edge_index_dict.items():
            if src_type == node_type and dst_type != node_type:
                forward_edges.append((edge_index, dst_type, "forward"))
            if dst_type == node_type and src_type != node_type:
                # Reverse traversal: we can walk *backward* along this edge
                forward_edges.append((edge_index.flip(0), src_type, "backward"))

        if not forward_edges:
            raise ValueError(f"No edges found leaving node type '{node_type}'.")

        # Build adjacency dicts for fast neighbour lookup.
        # adj_out[other_type] : dict[int -> Tensor]  node_type node -> neighbours
        # adj_back[other_type]: dict[int -> Tensor]  other_type node -> node_type nodes
        adj_out: dict[str, dict] = {}
        adj_back: dict[str, dict] = {}

        for (src_type, rel, dst_type), edge_index in data.edge_index_dict.items():
            srcs, dsts = edge_index[0], edge_index[1]

            if src_type == node_type and dst_type != node_type:
                if dst_type not in adj_out:
                    adj_out[dst_type] = {}
                for s, d in zip(srcs.tolist(), dsts.tolist()):
                    adj_out[dst_type].setdefault(s, []).append(d)
                # Return hop: from dst_type back to node_type
                if dst_type not in adj_back:
                    adj_back[dst_type] = {}
                for s, d in zip(dsts.tolist(), srcs.tolist()):
                    adj_back[dst_type].setdefault(s, []).append(d)

            if dst_type == node_type and src_type != node_type:
                # Can also enter via reverse: node_type <- other_type
                if src_type not in adj_out:
                    adj_out[src_type] = {}
                for s, d in zip(dsts.tolist(), srcs.tolist()):
                    adj_out[src_type].setdefault(s, []).append(d)
                if src_type not in adj_back:
                    adj_back[src_type] = {}
                for s, d in zip(srcs.tolist(), dsts.tolist()):
                    adj_back[src_type].setdefault(s, []).append(d)

        dest_nodes = torch.full((total,), -1, dtype=torch.long)
        inter_nodes = torch.full((total,), -1, dtype=torch.long)

        other_types = list(set(adj_out.keys()) & set(adj_back.keys()))
        if not other_types:
            raise ValueError(
                f"No round-trip paths found for node type '{node_type}'."
            )

        for i, src in enumerate(sources.tolist()):
            # --- Hop 1: pick a random intermediate type, then a random neighbour ---
            # Gather all reachable intermediate nodes across all relation types.
            candidates: list[tuple[str, int]] = []
            for ot in other_types:
                if src in adj_out[ot]:
                    for nb in adj_out[ot][src]:
                        candidates.append((ot, nb))

            if not candidates:
                continue  # dead end — stays -1

            rand_idx = torch.randint(len(candidates), (1,), generator=generator).item()
            inter_type, inter_node = candidates[rand_idx]
            inter_nodes[i] = inter_node

            # --- Hop 2: from intermediate node back to node_type ---
            return_candidates = adj_back[inter_type].get(inter_node, [])
            if not return_candidates:
                continue  # dead end on return

            rand_idx2 = torch.randint(len(return_candidates), (1,), generator=generator).item()
            dest_nodes[i] = return_candidates[rand_idx2]

        walks = torch.stack([sources, dest_nodes], dim=1)  # (total, 2)

        if return_intermediate:
            return walks, inter_nodes
        return walks


def filter_valid_walks(walks: torch.Tensor) -> torch.Tensor:
    """Remove walks that hit a dead end (destination == -1)."""
    return walks[walks[:, 1] != -1]
# ```

# **How it works for your schema:**

# The walk follows a two-hop pattern. Starting from, say, an `artist` node, hop 1 traverses *any* edge that connects `artist` to another type — so it can land on either a `performance` (via `performs`) or a `song` (via `composes`). Hop 2 then traverses back along any edge that returns to `artist`. The schema's bidirectional relations make this natural:
# ```
# artist → (performs) → performance → (performs, reversed) → artist
# artist → (composes) → song        → (composes, reversed) → artist
# artist → (performs) → performance → (performing, reversed) → ...
#   [dead end — performance→song has no return to artist]