from collections import deque
from torch_geometric.data import HeteroData
import torch

def bfs_hetero_with_depth(data: HeteroData, start_node_id=None, start_node_type=None, max_depth=None):
    """
    BFS with depth information.

    Yields:
        Tuple of (node_id, node_type, depth)
    """
    # Determine starting node
    if start_node_id is not None and start_node_type is not None:
        start = (start_node_id, start_node_type)
    else:
        start_node_type = data.node_types[0]
        start_node_id = 0
        start = (start_node_id, start_node_type)

    # Initialize BFS with depth tracking
    visited = set()
    queue = deque([(start, 0)])  # (node, depth)
    visited.add(start)

    # Build adjacency
    adjacency = {}
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        edge_index = data[edge_type].edge_index

        for i in range(edge_index.size(1)):
            src_id = edge_index[0, i].item()
            dst_id = edge_index[1, i].item()

            src_key = (src_id, src_type)
            if src_key not in adjacency:
                adjacency[src_key] = []
            adjacency[src_key].append((dst_id, dst_type))

    # BFS
    while queue:
        (node_id, node_type), depth = queue.popleft()

        # Stop if max depth reached
        if max_depth is not None and depth > max_depth:
            continue

        yield (node_id, node_type, depth)

        # Get neighbors
        node_key = (node_id, node_type)
        neighbors = adjacency.get(node_key, [])

        for neighbor_id, neighbor_type in neighbors:
            neighbor = (neighbor_id, neighbor_type)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
