import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GATv2Conv

from utils import *

class CGCL(nn.Module):
    def __init__(
            self,
            edge_decoder,
            dim_feature,
            dim_hidden_feature,
            dropout=0.8
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(dim_feature, dim_hidden_feature))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self.edge_decoder = edge_decoder

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.edge_decoder.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = to_sparse_tensor(edge_index, x.size(0))

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.activation(x)

        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.activation(x)

        return x
