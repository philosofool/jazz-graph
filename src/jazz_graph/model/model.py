from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, GATv2Conv

from jazz_graph.data.graph_builder.make_jazz import INDEXED_INSTRUMENTS, PerformanceFeatures

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData
    from collections.abc import Sequence

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
        if hasattr(batch['performance'], 'n_id'):
            x_dict = {
                'performance': self.performance_embed(batch['performance'].n_id),
                'artist': self.artist_embed(batch['artist'].n_id),
                'song': self.song_embed(batch['song'].n_id)
            }
        else:
            x_dict = {
                'performance': self.performance_embed(torch.arange(batch['performance'].num_nodes)),
                'artist': self.artist_embed(torch.arange(batch['artist'].num_nodes)),
                'song': self.song_embed(torch.arange(batch['song'].num_nodes))
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
    def __init__(self, gnn_encoder: nn.Module, embeddings_dim, projection_dim):
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
        model_config = config['model'].copy()
        metadata = (
            ['performance', 'song', 'artist'],
            [
                ('artist', 'composed', 'song'),
                ('artist', 'performs', 'performance'),
                ('performance', 'performing', 'song'),
                ('song', 'rev_composed', 'artist'),
                ('performance', 'rev_performs', 'artist'),
                ('song', 'rev_performing', 'performance'),
            ])
        model_config['metadata'] = metadata

        # This method is a mess. Too much happening here.
        # Couples config to method but config is not consistent. Ick.
        JazzModelClass = cls._model_class_from_config(config)

        def extract_edge_dim(value):
            return {tuple(e[0]): e[1] for e in value}

        if JazzModelClass is JazzModelWithStylesAndEdges:
            projection_dim = model_config['projection_dim']
            model_projection_dim  = model_config.pop('style_projection_dim')
            model_config['projection_dim'] = model_projection_dim
            model_config['edge_dim'] = extract_edge_dim(model_config['edge_dim'])
        else:
            projection_dim = model_config.pop('projection_dim')
        return cls(
            JazzModelClass(
                **model_config
            ),
            embeddings_dim=model_config['hidden_dim'],
            projection_dim=projection_dim
        )

    @classmethod
    def _model_class_from_config(cls, config):
        graph_function = config.get('graph_data_function')
        if graph_function is None:
            return JazzModel
        if graph_function == 'make_jazz_graph_with_style_and_edges':
            return JazzModelWithStylesAndEdges
        raise TypeError(f"{graph_function} is not supported in from_config.")

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


class GNNModelWithStylesAndEdges(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            input_dim: dict[str, int],
            metadata: tuple,
            edge_dim: dict[tuple[str, str, str], int],
            model_type: str | Sequence[str] = 'gatv2',
            num_layers=3,
            dropout=0.
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Create layers dynamically
        self.convs = nn.ModuleList()
        edge_types = metadata[1]
        if isinstance(model_type, str):
            model_type = [model_type for _ in range(len(edge_types))]

        if len(model_type) != len(edge_types):
            raise ValueError("When passed as sequence, there should be one model type per edge type.")

        for layer_num in range(num_layers):
            hetero_convs = {}
            for j, edge_type in enumerate(edge_types):
                in_dim = self._input_dims(layer_num, model_type[j], edge_type)
                hetero_convs[edge_type] = self._model_type(model_type[j], in_dim, hidden_dim, edge_dim.get(edge_type))
            self.convs.append(HeteroConv(hetero_convs))

    def _input_dims(self, layer_num, model_type, edge_type):
        if layer_num == 0:
            src = self.input_dim[edge_type[0]]
            dst = self.input_dim[edge_type[-1]]
            return (src, dst) if model_type == 'gatv2' else src
        return (self.hidden_dim, self.hidden_dim) if model_type == 'gatv2' else self.hidden_dim


    def _model_type(self, model_type, in_dim, hidden_dim, edge_dim):
        if model_type == 'sage':
            if edge_dim is not None:
                ...
                # raise ValueError("SAGEConv is not supported with edge attributes.")
            return SAGEConv(-1, hidden_dim, normalize=True)
        elif model_type == 'gat':
            return GATConv(in_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        elif model_type == 'gatv2':
            return GATv2Conv(in_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        else:
            raise ValueError("Expected model type 'sage', 'gat' or 'gatv2'.")

    def forward(self, x_dict, edge_dict, edge_attrs):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_dict, edge_attrs)

            # Apply ReLU + dropout to all layers except the last
            if i < self.num_layers - 1:
                x_dict = {key: F.normalize(self.dropout(F.relu(val)), dim=-1) for key, val in x_dict.items()}
            else:
                # Last layer: dropout only (no ReLU)
                x_dict = {key: F.normalize(self.dropout(val), dim=-1) for key, val in x_dict.items()}

        return x_dict

class JazzModelWithStylesAndEdges(nn.Module):
    def __init__(
        self,
        num_performances: int,
        num_artists: int,
        num_songs: int,
        embed_dim: int,
        hidden_dim: int,
        projection_dim: int,
        metadata: tuple,
        edge_dim: dict[tuple[str, str, str], int],
        dropout: float = 0.0,
        num_layers=3,
        model_type='sage'
    ):
        super().__init__()

        self.performance_embed = nn.Embedding(num_performances, embed_dim)
        self.song_embed = nn.Embedding(num_songs, embed_dim)
        self.artist_embed = nn.Embedding(num_artists, embed_dim)


        num_instruments = len(INDEXED_INSTRUMENTS)
        num_labels = len(PerformanceFeatures.style_columns)
        self.style_projection = nn.Linear(num_labels, projection_dim)
        self.instrument_embeddings = nn.Embedding(num_instruments, edge_dim[('artist', 'performs', 'performance')])


        node_dims = {'performance': projection_dim + embed_dim, 'artist': embed_dim, 'song': embed_dim}
        self.gnn = GNNModelWithStylesAndEdges(hidden_dim, node_dims, metadata, edge_dim, model_type, num_layers, dropout)

    def forward(self, x_dict, edge_dict, batch: HeteroData) -> torch.Tensor:
        if hasattr(batch['performance'], 'n_id'):
            x_dict = {
                'performance': self.performance_embed(batch['performance'].n_id),
                'artist': self.artist_embed(batch['artist'].n_id),
                'song': self.song_embed(batch['song'].n_id)
            }
        else:
            x_dict = {
                'performance': self.performance_embed(torch.arange(batch['performance'].num_nodes)),
                'artist': self.artist_embed(torch.arange(batch['artist'].num_nodes)),
                'song': self.song_embed(torch.arange(batch['song'].num_nodes))
            }
        style_embed = self.style_projection(batch['performance'].style)
        perfomance_embeds = x_dict['performance']
        x_dict['performance'] = torch.concat([perfomance_embeds, style_embed], dim=-1)

        edge_instruments = batch['artist', 'performs', 'performance'].edge_attr.squeeze()
        instrument_embeddings = self.instrument_embeddings(edge_instruments)
        edge_instruments_rev = batch['performance', 'rev_performs', 'artist'].edge_attr.squeeze()
        instrument_embeddings_rev = self.instrument_embeddings(edge_instruments_rev)

        x_dict = self.gnn(x_dict, edge_dict, {
            ('artist', 'performs', 'performance'): instrument_embeddings,
            ('performance', 'rev_performs', 'artist'): instrument_embeddings_rev
        })
        return x_dict
