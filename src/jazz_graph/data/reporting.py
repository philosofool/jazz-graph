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