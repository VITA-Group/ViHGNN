# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph, construct_hyperedges
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class HypergraphConv2d(nn.Module):
    """
    Hypergraph Convolution based on the GIN mechanism
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(HypergraphConv2d, self).__init__()
        # Node to hyperedge transformation
        self.nn_node_to_hyperedge = BasicConv([in_channels, in_channels], act, norm, bias) # in_channels = 128, out_channels = 256
        # Hyperedge to node transformation
        self.nn_hyperedge_to_node = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, hyperedge_matrix, point_hyperedge_index, centers):
        with torch.no_grad():
            # Check and append dummy node to x if not present
            if not torch.equal(x[:, :, -1, :], torch.zeros((x.size(0), x.size(1), 1, x.size(3)), device=x.device)): # 假设 n_dims = 128, n_points = 3136, hyperedge_num = 50
                dummy_node = torch.zeros((x.size(0), x.size(1), 1, x.size(3)), device=x.device)
                x = torch.cat([x, dummy_node], dim=2) # (1, 128, 3137, 1)
            
            # Check and append dummy hyperedge to centers if not present
            if not torch.equal(centers[:, :, -1], torch.zeros((centers.size(0), centers.size(1), 1), device=centers.device)):
                dummy_hyperedge = torch.zeros((centers.size(0), centers.size(1), 1), device=centers.device)
                centers = torch.cat([centers, dummy_hyperedge], dim=2) # centers: (1, 128, 51)

        # Step 1: Aggregate node features to get hyperedge features
        node_features_for_hyperedges = batched_index_select(x, hyperedge_matrix)
        aggregated_hyperedge_features, _ = node_features_for_hyperedges.sum(dim=-1, keepdim=True)
        aggregated_hyperedge_features = self.nn_node_to_hyperedge(aggregated_hyperedge_features.squeeze(-1)) # (1, 128, 50)
        # Adding the hyperedge center features to the aggregated hyperedge features
        aggregated_hyperedge_features += (1 + self.eps) * centers[:, :, :-1, :]
        
        # Step 2: Aggregate hyperedge features to update node features
        hyperedge_features_for_nodes = batched_index_select(aggregated_hyperedge_features.unsqueeze(-1), point_hyperedge_index)
        aggregated_node_features_from_hyperedges = self.nn_hyperedge_to_node(hyperedge_features_for_nodes.sum(dim=-1, keepdim=True).squeeze(-1))

        # Update original node features
        out = aggregated_node_features_from_hyperedges

        return out


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'hypergraph':
            self.gconv = HypergraphConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None, hyperedge_matrix=None, point_hyperedge_index=None, centers=None):
        if isinstance(self.gconv, HypergraphConv2d):
            return self.gconv(x, edge_index, y, hyperedge_matrix=hyperedge_matrix, point_hyperedge_index=point_hyperedge_index, centers=centers)
        else:
            return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        # Choose between dilated knn graph and hypergraph
        if conv == 'hypergraph':
            self.use_hypergraph = True
        else:
            self.use_hypergraph = False
            self.graph_constructor = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        x = x.reshape(B, C, -1, 1).contiguous()
        
        # Construct graph using either hypergraph or dilated knn graph based on use_hypergraph flag
        if self.use_hypergraph:
            hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(x, num_clusters=self.k)
            x = super(DyGraphConv2d, self).forward(x, hyperedge_matrix=hyperedge_matrix, point_hyperedge_index=point_hyperedge_index, centers=centers, y=y)
        else:
            edge_index = self.graph_constructor(x, y, relative_pos)
            x = super(DyGraphConv2d, self).forward(x, edge_index, y=y)
        
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
