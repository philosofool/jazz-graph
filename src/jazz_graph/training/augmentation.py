# Definte augmentation Strategies

from jazz_graph.data.graph_transforms import drop_edge_from_masks, prune_graph_from_masks


from torch_geometric.data import HeteroData


def drop_edge_augmentation(graph: HeteroData, dst_graph, drop_edge_prob: float = .2):
    edge_types = graph.metadata()[1]
    edge_masks = {
        edge_type: torch.rand(graph[edge_type].edge_index.size(1)) > drop_edge_prob
        for edge_type in edge_types
    }
    return drop_edge_from_masks(graph, edge_masks, dst_graph)


def drop_random_nodes_and_edges(data: HeteroData, drop_edge_prob: float = .5):
    out = data.clone()
    drop_edge_augmentation(data, out, drop_edge_prob=drop_edge_prob)
    # drop_node_augmentation(data, out)  # This would be complex: graphs need to align their node indecies in loss.
    return out


def drop_node_augmentation(src_graph: HeteroData, dst_graph: HeteroData, drop_node_prob: float = .1):
    node_types, edge_types = src_graph.metadata()
    masks = {
        node_type: torch.rand(src_graph[node_type].num_nodes) > drop_node_prob
        for node_type in node_types
    }
    prune_graph_from_masks(src_graph, masks)