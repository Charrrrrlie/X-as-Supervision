import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, norm

from torch import Tensor
from typing import Tuple

def my_batched_dense_to_sparse(
    adj: Tensor
) -> Tuple[Tensor, Tensor]:
    # use out-place operation
    # source code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/sparse.html

    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be two- or "
                         f"three-dimensional (got {adj.dim()} dimensions)")

    if adj.dim() == 2:
        edge_index = adj.nonzero().t()
        edge_attr = adj[edge_index[0].clone(), edge_index[1].clone()]
        return edge_index, edge_attr
    else:
        flatten_adj = adj.view(-1, adj.size(-1))

        edge_index = flatten_adj.nonzero().t()
        edge_attr = flatten_adj[edge_index[0].clone(), edge_index[1].clone()]

        offset = torch.arange(
            start=0,
            end=adj.size(0) * adj.size(2),
            step=adj.size(2),
            device=adj.device,
        )
        offset = offset.repeat_interleave(adj.size(1))

        edge_index[1] += offset[edge_index[0]]

        return edge_index, edge_attr

class GCN_simple(torch.nn.Module):
    def __init__(self, input_dim, output_dim, self_loop=False):
        super(GCN_simple, self).__init__()
        self.gc = GCNConv(input_dim, output_dim, add_self_loops=self_loop)

    def forward(self, input):
        x, adj, edge_weights = input
        x = self.gc(x, adj, edge_weights)
        x = F.relu(x)
        return (x, adj, edge_weights)


class GCN_residual(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, self_loop=False, use_bn=False, p_dropout=0.5):
        super(GCN_residual, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim, add_self_loops=self_loop)
        self.gc2 = GCNConv(hidden_dim, output_dim, add_self_loops=self_loop)
        self.use_bn = use_bn
        if use_bn:
            self.bn = torch.nn.SyncBatchNorm(output_dim)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(self, input):
        x, adj, edge_weights = input
        res = x

        x = self.gc1(x, adj, edge_weights)
        if self.use_bn:
            x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gc2(x, adj, edge_weights)
        if self.use_bn:
            x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return (x + res, adj, edge_weights)

class GCN_SAGE_residual(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, single_layer=False):
        super(GCN_SAGE_residual, self).__init__()
        self.single_layer = single_layer
        if not self.single_layer:
            self.gc1 = SAGEConv(input_dim, hidden_dim, aggr='mean')
            self.ln1 = norm.LayerNorm(hidden_dim)
            self.gc2 = SAGEConv(hidden_dim, output_dim, aggr='mean')
            self.ln2 = norm.LayerNorm(output_dim)
        else:
            self.gc1 = SAGEConv(input_dim, output_dim, aggr='mean')
            self.ln1 = norm.LayerNorm(output_dim)
            self.gc2 = torch.nn.Identity()

    def forward(self, input):
        x, adj = input
        if self.single_layer:
            x = self.gc1(x, adj)
            x = self.ln1(x)
            x = F.relu(x)
            return (x, adj)

        else:
            res = x
            x = self.gc1(x, adj)
            x = self.ln1(x)
            x = F.relu(x)
            x = self.gc2(x, adj)
            x = self.ln2(x)
            x = F.relu(x)

            return (x + res, adj)

if __name__ == '__main__':
    adj = torch.tensor([[[3, 1], [2, 0]], [[0, 1], [0, 2]]])
    # adj = torch.tensor([[[3, 1, 0], [2, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 2, 3], [0, 5, 0]]])
    edge_index, edge_weight = my_batched_dense_to_sparse(adj)
    print(edge_index, edge_weight)