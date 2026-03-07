from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

class GNNModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, metadata, model_type='sage', num_layers=3, dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Create layers dynamically
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(
                HeteroConv({
                    key: self._model_type(model_type, in_dim, hidden_dim) for key in metadata[1]
                })
            )

    def _model_type(self, model_type, in_dim, hidden_dim):
        if model_type == 'sage':
            return SAGEConv(in_dim, hidden_dim, normalize=True)
        elif model_type == 'gat':
            return GATConv(in_dim, hidden_dim, add_self_loops=False)
        else:
            raise ValueError("Expected model type 'sage' or 'gat'.")

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
        num_layers=3,
        model_type='sage'
    ):
        super().__init__()

        self.performance_embed = nn.Embedding(num_performances, embed_dim)
        self.song_embed = nn.Embedding(num_songs, embed_dim)
        self.artist_embed = nn.Embedding(num_artists, embed_dim)

        self.gnn = GNNModel(hidden_dim, embed_dim, metadata, model_type, num_layers, dropout)


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
    def __init__(self, base_model: JazzModel, target_edge: tuple[str, str, str]):
        super().__init__()
        self.base_model = base_model
        self.target_edge = target_edge

    def forward(self, batch) -> torch.Tensor:
        src, relation, dst = self.target_edge
        x_dict, edge_index_dict, edge_label_index = batch.x_dict, batch.edge_index_dict, batch[relation].edge_label_index
        x_dict = self.base_model(x_dict, edge_index_dict, batch)
        # Do dot-product classification.
        #   Align the artist learned feature by the edge index, same for performance:
        src_to_dst = x_dict[src][edge_label_index[0]]
        #   Same, for performances:
        dst_to_src = x_dict[dst][edge_label_index[1]]
        #   Compute logits as dot-product
        logits = (src_to_dst * dst_to_src).sum(-1)
        return logits


# Define a model

class UnsupervisedJazzModel(nn.Module):
    """SimCLR style self-supervised learning model for jazz graphs."""
    def __init__(self, gnn_encoder: JazzModel, embeddings_dim, projection_dim):
        super().__init__()
        self.base_model = gnn_encoder
        node_types = ['performance', 'artist', 'song']
        self.projections = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(embeddings_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )
            for key in node_types
        })
        self.layer_normalization = nn.ModuleDict({
            node_type: nn.LayerNorm(embeddings_dim) for node_type in ['performance', 'artist', 'song']
        })

    @classmethod
    def from_config(cls, config: dict):
        model_config = config['model']
        metadata = (
            ['performance', 'song', 'artist'],
            [
                ('artist', 'performs', 'performance'),
                ('performance', 'performing', 'song'),
                ('artist', 'composed', 'song'),
                ('performance', 'rev_performs', 'artist'),
                ('song', 'rev_performing', 'performance'),
                ('song', 'rev_composed', 'artist')])
        return cls(
            JazzModel(
                num_performances=model_config['num_performances'],
                num_artists=model_config['num_artists'],
                num_songs=model_config['num_songs'],
                hidden_dim=model_config['hidden_dim'],
                embed_dim=model_config['embed_dim'],
                dropout=model_config['dropout'],
                metadata=metadata,
                num_layers=model_config['num_layers'],
                model_type=model_config['model_type']
            ),
            embeddings_dim=model_config['hidden_dim'],
            projection_dim=model_config['projection_dim']
        )

    def encode(self, batch):
        x_dict = self.base_model(batch.x_dict, batch.edge_index_dict, batch)
        for node_type, layer_norm in self.layer_normalization.items():
            x_dict[node_type] = layer_norm(x_dict[node_type])
        return x_dict

    def project(self, x_dict):
        for key, projection in self.projections.items():
            x_dict[key] = projection(x_dict[key])
        x_dict = {key: F.normalize(tensor, dim=-1) for key, tensor in x_dict.items()}
        return x_dict

    def forward(self, batch):
        x_dict = self.encode(batch)
        x_dict = self.project(x_dict)
        return x_dict