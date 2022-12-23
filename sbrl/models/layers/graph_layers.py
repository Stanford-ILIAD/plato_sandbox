import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from sbrl.utils.python_utils import get_required, get_with_default


class GCNConv(MessagePassing):
    def __init__(self, params):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        # takes (N, in_channels) --> (N, out_channels)
        self.feature_map = get_required(params, "map").to_module_list()
        # if true, will flatten before calling propogate, and then unflatten after
        self.flatten_dims = get_with_default(params, "flatten_dims", True)

    def forward(self, x_and_edge_idx):
        x, edge_index = x_and_edge_idx
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.feature_map(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        shp = x.shape

        if self.flatten_dims:
            x = x.view(x.shape[0], -1)

        new_x, new_edge_idx = self.propagate(edge_index, x=x, norm=norm), edge_index

        if self.flatten_dims:
            new_x = new_x.view(shp)

        return new_x, new_edge_idx

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return torch.multiply(norm.reshape(-1, 1), x_j)


class EdgeConv(MessagePassing):
    def __init__(self, params):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.edge_map = get_required(params, "map").to_module_list()
        # if true, will flatten before calling propogate, and then unflatten after
        self.flatten_dims = get_with_default(params, "flatten_dims", True)

    def forward(self, x_and_edge_idx):
        x, edge_index = x_and_edge_idx
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        shp = x.shape
        if self.flatten_dims:
            x = x.view(x.shape[0], -1)

        new_x, new_edge_idx = self.propagate(edge_index, x=x, shp=shp), edge_index

        if self.flatten_dims:
            new_x = new_x.view(*shp[:-1], -1)  # last dim will be out channels

        return new_x, new_edge_idx

    def message(self, x_i, x_j, shp):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # shp is a torch shape
        if self.flatten_dims:
            x_i = x_i.view(x_i.shape[0], *shp[1:])
            x_j = x_j.view(x_j.shape[0], *shp[1:])

        tmp = torch.cat([x_i, x_j - x_i], dim=-1)  # tmp has shape [E, ..., 2*in_channels]

        edge_out = self.edge_map(tmp)  # returns [E, ..., out_channels], which then get aggr'd

        if self.flatten_dims:
            edge_out = edge_out.view(edge_out.shape[0], -1)

        return edge_out