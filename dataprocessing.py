import torch
from torch_geometric.datasets import Amazon, Planetoid


def generate_two_views(edge_index):
    p = 1/2
    edge_views = list()
    edge_index_set = torch.arange(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    edge_index_vector = torch.full_like(edge_index_set, p, dtype=torch.float32)
    mask = torch.bernoulli(edge_index_vector).to(torch.bool)
    edge_views.append(edge_index[:, ~mask])
    edge_views.append(edge_index[:, mask])

    return edge_views


def load_data(dataset):
    root = 'data/'
    if dataset in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, dataset)
    elif dataset in {'Photo', 'Computers'}:
        dataset = Amazon(root, dataset)
    else:
        raise ValueError(dataset)

    return dataset



