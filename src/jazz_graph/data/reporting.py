from __future__ import annotations
import pandas as pd
from torch_geometric.utils import degree
import numpy as np
import random


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

def inspect_degrees(data: HeteroData, percentiles=None) -> pd.DataFrame:
    """Compute degree distribution statistics for all edge types in a heterogeneous graph.

    For each edge type (relation), computes the out-degree distribution of source nodes
    (i.e., how many outgoing edges each source node has). Returns summary statistics
    including count, mean, std, min, quartiles, and max.

    Args:
        data: PyTorch Geometric HeteroData object containing the graph
        percentiles: Optional list of percentiles to include (e.g., [.25, .5, .75, .9, .95]).
                    If None, uses pandas default percentiles [.25, .5, .75]

    Returns:
        DataFrame where:
            - Columns are edge type names (relations)
            - Rows are statistics (count, mean, std, min, 25%, 50%, 75%, max, etc.)
            - Values represent the distribution of out-degrees for source nodes

    Example:
        >>> stats = inspect_degrees(data, percentiles=[.5, .9, .95, .99])
        >>> print(stats)
                     performs  composes  performance_of
        count      2583.00   2583.00          10886.00
        mean         12.65      1.70              1.00
        std          24.31      2.15              0.03
        min           0.00      0.00              1.00
        50%           8.00      1.00              1.00
        90%          46.00      4.00              1.00
        95%          67.00      6.00              1.00
        99%         128.00     10.00              1.00
        max         500.00     42.00              2.00

    Note:
        Out-degree counts edges FROM source nodes. For 'artist -[performs]-> performance',
        this shows how many performances each artist participated in.
    """
    out = {}
    for a, relation, b in data.metadata()[1]:
        n_nodes = data[a].num_nodes
        edge_index = data[(a, relation, b)].edge_index[0]
        node_relation_degrees = degree(edge_index, n_nodes).numpy()
        out[relation] = pd.Series(node_relation_degrees).describe(percentiles).values
    index = pd.Series(node_relation_degrees).describe(percentiles).index
    return pd.DataFrame(out, index=index)


class EdgeCharacteristics:
    def __init__(self, edge_characteristics: pd.DataFrame, relations: dict[str, dict[int, int]]):
        self.relations = relations
        self.edge_characteristics = edge_characteristics

    def describe_edge(self, src_node: int, edge_type: tuple[str, str, str], dst_node: int):
        relation = edge_type[1]
        src_idx = self.source_to_edge(src_node, relation)
        dst_idx = self.source_to_edge(dst_node, relation)
        edge_characteristics = self.fetch_edge_characteristics(edge_type)
        return edge_characteristics.loc[(src_idx, dst_idx)]

    def fetch_edge_characteristics(self, edge_type: tuple[str, str, str]) -> pd.DataFrame:
        src, edge, dst = edge_type
        src_data = self.get_node_data(src)
        dst_data = self.get_node_data(dst)
        edge_map = self.get_edge_data(edge)
        return edge_map.merge(src_data).merge(dst_data).set_index([src, dst])

    def source_to_edge(self, node: int, relation: str) -> int:
        edge_index_to_node_id = self.relations[relation]
        return edge_index_to_node_id[node]


if __name__ == '__main__':
    relations = {'performs': [[0, 1], [0, 20]], 'composes': [[10, 30]]}
    edge_characteristics =pd.DataFrame.from_records([
        (11, 'performs', 20, 'McCoy Tyner', 'Psalm'),
        (10, 'performs', 20, 'John Coltrane', 'Psalm'),
        (10, 'composes', 30, "John Coltrane", "Psalm")
    ])

    # Are val supervision edges present in the message passing graph?
    val_supervise = set(map(tuple, dev_data['artist', 'performs', 'performance'].edge_label_index.t().tolist()))
    val_msg_passing = set(map(tuple, dev_data['artist', 'performs', 'performance'].edge_index.t().tolist()))

    leakage = val_supervise & val_msg_passing
    print(f"Supervision edges also in message passing: {len(leakage)} / {len(val_supervise)}")

    # Check reverse edges
    val_reverse = set(map(tuple, dev_data['performance', 'rev_performs', 'artist'].edge_index.t().tolist()))
    val_supervise_reversed = {(b, a) for a, b in val_supervise}
    reverse_leakage = val_supervise_reversed & val_reverse
    assert not reverse_leakage, f"Supervision edges leaked via reverse edges: {len(reverse_leakage)} / {len(val_supervise)}"

    train_supervise = set(map(tuple, train_data['artist', 'performs', 'performance'].edge_label_index.t().tolist()))
    train_msg_passing = set(map(tuple, train_data['artist', 'performs', 'performance'].edge_index.t().tolist()))
    train_leakage = train_supervise & train_msg_passing

    train_labels = train_data['artist', 'performs', 'performance'].edge_label
    train_num_positive = (train_labels == 1).sum().item()

    print(f"TRAIN:")
    print(f"  Positive samples: {train_num_positive}")
    print(f"  Edges in both: {len(train_leakage)}")
    print(f"  Percentage: {len(train_leakage) / train_num_positive * 100:.1f}%")

    val_labels = dev_data['artist', 'performs', 'performance'].edge_label
    num_positive = (val_labels == 1).sum().item()
    num_negative = (val_labels == 0).sum().item()

    print(f"Positive samples: {num_positive}")
    print(f"Negative samples: {num_negative}")
    print(f"Edges in both: ")
    print(f"Percentage of positives in message passing: {175 / num_positive * 100:.1f}%")


import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# def get_node_neigborhood(graph: HeteroData, node: int, node_type: str) -> dict[str, list]:
#     edges = graph.metadata()[1]
#     nodes = graph.metadata()[0]
#     neighborhood = defaultdict(list)
#     for edge in edges:
#         if node_type == edge[0]:
#             neighbortype = edge[2]
#             neighbor_mask = graph[edge].edge_index[0] == node
#             neighbors = graph[edge].edge_index[1][neighbor_mask]
#             neighborhood[neighbortype].extend(neighbors.tolist())
#         elif node_type == edge[2]:
#             neighbortype = edge[0]
#             neighbor_mask = graph[edge].edge_index[1] == node
#             neighbors = graph[edge].edge_index[0][neighbor_mask]
#             neighborhood[neighbortype].extend(neighbors.tolist())
#     return neighborhood

# def sample_node_neighborhood(neighborhood: dict, n_sample: int | dict[str, int]):
#     if isinstance(n_sample, int):
#         n_sample = {key: n_sample for key in neighborhood.keys()}
#     out = {}
#     for key, values in neighborhood.items():
#         n = n_sample[key]
#         if n < len(values):
#             sampled_values = values
#         else:
#             sampled_values = np.random.choice(values, n, replace=False)
#         out[key] = sampled_values
#     return out


def sample_graph(graph: HeteroData, from_node_type: str, threshold=128):
    """A simple graph sampling function.

    Algorithm:
    1. Get a single node of the specified type.
    2. Get the immediate neighborhood of that node.
    3. If the total number of nodes exceeds threshold, stop.
       Otherwise, return to 1.
    """
    edge_types = graph.metadata()[1]
    node_types = graph.metadata()[0]
    if from_node_type not in node_types:
        raise ValueError("Node type must be in graph metadata.")
    candidate_nodes: dict[str, list] = {node_type: list(range(graph[node_type].num_nodes)) for node_type in node_types}
    next_node_idx = random.randint(0, len(candidate_nodes[from_node_type]) - 1)
    next_node = candidate_nodes[from_node_type].pop(next_node_idx)
    next_neighborhood = get_node_neigborhood(graph, next_node, from_node_type)
    neighborhoods = [next_neighborhood]
    while neighborhoods:
        next_neighborhood = neighborhoods.pop(0)
        for key, value in next_neighborhood.items():
            for node in value:
                neighborhoods.append(get)



# Credit: Claude.ai
def visualize_hetero_graph(data, sample_nodes=None, figsize=(15, 10), show_edge_labels=True):
    """
    Visualize a PyTorch Geometric HeteroData graph.

    Arguments
    ---------
        data: HeteroData object
        sample_nodes: Dict mapping node types to number of nodes to sample
                     e.g., {'artist': 20, 'performance': 30, 'song': 15}
                     If None, samples 10 of each type
        figsize: Figure size tuple
        show_edge_labels: Whether to show edge type labels

    Usage
    -----

    fig, ax = visualize_hetero_graph(
        data,
        sample_nodes={'artist': 15, 'performance': 20, 'song': 10},
        figsize=(16, 12)
    )
    plt.show()
    """

    # Default sampling
    if sample_nodes is None:
        sample_nodes = {node_type: min(10, data[node_type].num_nodes)
                       for node_type in data.node_types}

    # Sample nodes
    sampled_nodes = {}
    for node_type, n_sample in sample_nodes.items():
        num_nodes = data[node_type].num_nodes
        n_sample = min(n_sample, num_nodes)
        sampled_nodes[node_type] = torch.randperm(num_nodes)[:n_sample].tolist()

    # Create NetworkX graph
    G = nx.MultiDiGraph()

    # Add nodes with type labels
    node_colors = {}
    color_map = {'artist': 'lightblue', 'performance': 'lightgreen',
                 'song': 'lightcoral', 'album': 'lightyellow'}

    pos = {}
    node_to_label = {}

    # Add sampled nodes
    for node_type, node_ids in sampled_nodes.items():
        for node_id in node_ids:
            # Create unique node identifier
            node_key = f"{node_type}_{node_id}"
            G.add_node(node_key, node_type=node_type)
            node_colors[node_key] = color_map.get(node_type, 'lightgray')
            node_to_label[node_key] = f"{node_type[0].upper()}{node_id}"

    # Add edges
    edge_counts = defaultdict(int)
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        edge_index = data[edge_type].edge_index

        # Filter edges to only sampled nodes
        for i in range(edge_index.size(1)):
            src_id = edge_index[0, i].item()
            dst_id = edge_index[1, i].item()

            if src_id in sampled_nodes[src_type] and dst_id in sampled_nodes[dst_type]:
                src_key = f"{src_type}_{src_id}"
                dst_key = f"{dst_type}_{dst_id}"
                G.add_edge(src_key, dst_key, relation=relation)
                edge_counts[relation] += 1

    # Create layout
    # Group nodes by type for better visualization
    node_types_list = list(sampled_nodes.keys())
    n_types = len(node_types_list)

    for i, node_type in enumerate(node_types_list):
        nodes = sampled_nodes[node_type]
        # Position nodes in columns by type
        x = i * 3  # Spread types horizontally
        for j, node_id in enumerate(nodes):
            node_key = f"{node_type}_{node_id}"
            y = j * 0.5  # Spread nodes vertically
            pos[node_key] = (x, y)

    # Draw
    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    colors = [node_colors[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, alpha=0.9, ax=ax)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=node_to_label, font_size=8, ax=ax)

    # Draw edges by relation type
    edge_colors = {
        'performs': 'blue',
        'composed': 'green',
        'performing': 'orange',
        'same_album': 'purple',
        'rev_performs': 'lightblue',
        'rev_composed': 'lightgreen',
        'rev_performing': 'wheat'
    }

    for relation, color in edge_colors.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == relation]
        if edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges,
                edge_color=color, alpha=0.5,
                arrows=True, arrowsize=10,
                label=f"{relation} ({len(edges)})",
                ax=ax
            )

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Print statistics
    print("Graph Statistics:")
    print(f"  Nodes sampled: {G.number_of_nodes()}")
    print(f"  Edges sampled: {G.number_of_edges()}")
    print(f"\nEdge counts by type:")
    for relation, count in sorted(edge_counts.items()):
        print(f"  {relation}: {count}")

    plt.title("Heterogeneous Graph Visualization", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    return fig, ax
