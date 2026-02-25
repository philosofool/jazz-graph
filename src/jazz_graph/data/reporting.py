from __future__ import annotations
import pandas as pd
from torch_geometric.utils import degree


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
    print(f"Edges in both: 175")
    print(f"Percentage of positives in message passing: {175 / num_positive * 100:.1f}%")