from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

class GNNModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, metadata, dropout=0.):
        super().__init__()
        self.conv1 = HeteroConv({
            key: SAGEConv(input_dim, hidden_dim) for key in metadata[1]
        })
        self.conv2 = HeteroConv({
            key: SAGEConv(hidden_dim, hidden_dim) for key in metadata[1]
        })
        self.conv3 = HeteroConv({
            key: SAGEConv(hidden_dim, hidden_dim) for key in metadata[1]
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_dict):
        x_dict = self.conv1(x_dict, edge_dict)
        x_dict = {key: self.dropout(F.relu(val)) for key, val in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_dict)
        x_dict = {key: self.dropout(F.relu(val)) for key, val in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_dict)
        x_dict = {key: self.dropout(val) for key, val in x_dict.items()}

        return x_dict

class JazzModel(nn.Module):
    def __init__(
        self,
        num_performances: int,
        num_artists: int,
        num_songs: int,
        embed_dim: int,
        hidden_dim: int,
        metadata: tuple,
        dropout: float = 0.0
    ):
        super().__init__()

        self.performance_embed = nn.Embedding(num_performances, embed_dim)
        self.song_embed = nn.Embedding(num_songs, embed_dim)
        self.artist_embed = nn.Embedding(num_artists, embed_dim)

        self.gnn = GNNModel(hidden_dim, embed_dim, metadata, dropout)


    def forward(self, x_dict, edge_dict) -> torch.Tensor:
        x_dict = {
            'performance': self.performance_embed(x_dict['performance'].view(-1)),
            'artist': self.artist_embed(x_dict['artist'].view(-1)),
            'song': self.song_embed(x_dict['song'].view(-1))
        }

        x_dict = self.gnn(x_dict, edge_dict)
        return x_dict


class NodeClassifier(nn.Module):
    def __init__(self, base_model, hidden_dim, num_classes):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.base_model(x_dict, edge_index_dict)
        logits = self.classifier(x_dict['performance'])
        return logits


class LinkPredictionModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x_dict, edge_index_dict, edge_label_index) -> torch.Tensor:
        x_dict = self.base_model(x_dict, edge_index_dict)
        # Do dot-product classification.
        #   Align the artist learned feature by the edge index, same for performance:
        artists_to_performance = x_dict['artist'][edge_label_index[0]]
        #   Same, for performances:
        performance_to_artist = x_dict['performance'][edge_label_index[1]]
        #   Compute logits as dot-product
        logits = (artists_to_performance * performance_to_artist).sum(-1)
        return logits