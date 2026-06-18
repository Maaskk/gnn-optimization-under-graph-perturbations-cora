import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network for Cora node classification."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

