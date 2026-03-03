from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

class GNNModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, metadata, num_layers=3, dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Create layers dynamically
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(
                HeteroConv({
                    key: SAGEConv(in_dim, hidden_dim, normalize=True) for key in metadata[1]
                })
            )

    def forward(self, x_dict, edge_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_dict)

            # Apply ReLU + dropout to all layers except the last
            if i < self.num_layers - 1:
                x_dict = {key: self.dropout(F.relu(val)) for key, val in x_dict.items()}
            else:
                # Last layer: dropout only (no ReLU)
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
        dropout: float = 0.0,
        num_layers=3
    ):
        super().__init__()

        self.performance_embed = nn.Embedding(num_performances, embed_dim)
        self.song_embed = nn.Embedding(num_songs, embed_dim)
        self.artist_embed = nn.Embedding(num_artists, embed_dim)

        self.gnn = GNNModel(hidden_dim, embed_dim, metadata, num_layers, dropout)


    def forward(self, x_dict, edge_dict, batch) -> torch.Tensor:
        x_dict = {
            'performance': self.performance_embed(batch['performance'].n_id),
            'artist': self.artist_embed(batch['artist'].n_id),
            'song': self.song_embed(batch['song'].n_id)
        }

        x_dict = self.gnn(x_dict, edge_dict)
        return x_dict


class NodeClassifier(nn.Module):
    def __init__(self, base_model: JazzModel, hidden_dim, num_classes):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
        x_dict = self.base_model(x_dict, edge_index_dict, batch)
        logits = self.classifier(x_dict['performance'])
        return logits


class LinkPredictionModel(nn.Module):
    def __init__(self, base_model: JazzModel):
        super().__init__()
        self.base_model = base_model

    def forward(self, batch) -> torch.Tensor:
        x_dict, edge_index_dict, edge_label_index = batch.x_dict, batch.edge_index_dict, batch['performs'].edge_label_index
        x_dict = self.base_model(x_dict, edge_index_dict, batch)
        # Do dot-product classification.
        #   Align the artist learned feature by the edge index, same for performance:
        artists_to_performance = x_dict['artist'][edge_label_index[0]]
        #   Same, for performances:
        performance_to_artist = x_dict['performance'][edge_label_index[1]]
        #   Compute logits as dot-product
        logits = (artists_to_performance * performance_to_artist).sum(-1)
        return logits