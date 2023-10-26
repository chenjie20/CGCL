import torch
import torch.nn as nn


class EdgeDecoder(nn.Module):
    def __init__(
            self, dim_hidden_feature_1, dropout=0.2):

        super().__init__()
        dim_output = 1
        dim_hidden_feature_2 = int(dim_hidden_feature_1/2)
        self.mul_layers = nn.ModuleList()
        self.mul_layers.append(nn.Linear(dim_hidden_feature_1, dim_hidden_feature_2))
        self.mul_layers.append(nn.Linear(dim_hidden_feature_2, dim_output))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def reset_parameters(self):
        for mul_layer in self.mul_layers:
            mul_layer.reset_parameters()

    def forward(self, hidden_features, edge, sigmoid=True):

        hidden_representation = hidden_features[edge[0]] * hidden_features[edge[1]]
        for i, mul_layer in enumerate(self.mul_layers[:-1]):
            hidden_representation = self.dropout(hidden_representation)
            hidden_representation = mul_layer(hidden_representation)
            hidden_representation = self.activation(hidden_representation)

        hidden_representation = self.mul_layers[-1](hidden_representation)

        if sigmoid:
            return hidden_representation.sigmoid()
        else:
            return hidden_representation
