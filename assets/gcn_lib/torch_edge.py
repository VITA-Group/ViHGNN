# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            ####
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        return self._dilated(edge_index)


def initialize_memberships(batch_size, n_points, n_clusters, device):
    """
    Initialize the membership matrix for Fuzzy C-Means clustering.

    Args:
        batch_size: int
        n_points: int
        n_clusters: int
        device: torch.device

    Returns:
        memberships: tensor (batch_size, n_points, n_clusters)
    """
    # Randomly initialize the membership matrix ensuring that the sum over clusters for each point is 1
    memberships = torch.rand(batch_size, n_points, n_clusters, device=device)
    memberships = memberships / memberships.sum(dim=2, keepdim=True)
    return memberships


def fuzzy_c_means(x, n_clusters, m=2, epsilon=1e-6, max_iter=1000):
    """
    Fuzzy C-Means clustering

    Args:
        x: tensor (batch_size, num_dims, num_points, 1)
        n_clusters: int, the number of clusters
        m: float, fuzziness parameter
        epsilon: float, threshold for stopping criterion
        max_iter: int, maximum number of iterations

    Returns:
        membership: tensor (batch_size, num_points, n_clusters)
        centers: tensor (batch_size, num_dims, n_clusters)
    """
    batch_size, num_dims, num_points, _ = x.size()
    x = x.squeeze(-1).transpose(1, 2)  # Shape: (batch_size, num_points, num_dims)

    # Initialize the membership matrix
    memberships = initialize_memberships(batch_size, num_points, n_clusters, x.device)

    # Initialize cluster centers
    centers = torch.zeros(batch_size, num_dims, n_clusters, device=x.device)
    prev_memberships = torch.zeros_like(memberships)

    for iteration in range(max_iter):
        # Update cluster centers
        for cluster in range(n_clusters):
            # Calculate the denominator
            weights = memberships[:, :, cluster] ** m
            denominator = weights.sum(dim=1, keepdim=True)
            # Update centers
            numerator = (weights.unsqueeze(2) * x).sum(dim=1)
            centers[:, :, cluster] = numerator / denominator

        # Update memberships
        for cluster in range(n_clusters):
            diff = x - centers[:, :, cluster].unsqueeze(1)
            dist = torch.norm(diff, p=2, dim=2)  # Euclidean distance
            memberships[:, :, cluster] = 1.0 / (dist ** (2 / (m - 1)))

        # Normalize the memberships such that each point's memberships across clusters sum to 1
        memberships_sum = memberships.sum(dim=2, keepdim=True)
        memberships = memberships / memberships_sum

        # Check convergence: stop if memberships do not change significantly
        if iteration > 0 and torch.norm(prev_memberships - memberships) < epsilon:
            break
        prev_memberships = memberships.clone()

    return memberships, centers


def construct_hyperedges(x, num_clusters, threshold=0.5, m=2):
    """
    Constructs hyperedges based on fuzzy c-means clustering.

    Args:
        x (torch.Tensor): Input point cloud data with shape (batch_size, num_dims, num_points, 1).
        num_clusters (int): Number of clusters (hyperedges).
        threshold (float): Threshold value for memberships to consider a point belonging to a cluster.
        m (float): Fuzzifier for fuzzy c-means clustering.

    Returns:
        hyperedge_matrix (torch.Tensor): Tensor of shape (batch_size, n_clusters, num_points_index).
            Represents each cluster's points. Padded with -1 for unequal cluster sizes.
        point_hyperedge_index (torch.Tensor): Tensor of shape (batch_size, num_points, cluster_index).
            Indicates the clusters each point belongs to. Padded with -1 for points belonging to different numbers of clusters.
        hyperedge_features (torch.Tensor): Tensor of shape (batch_size, num_dims, n_clusters).
            The center of each cluster, serving as the feature for each hyperedge.
    """
    
    with torch.no_grad():
        x = x.detach()  # Detach x from the computation graph
        
        batch_size, num_dims, num_points, _ = x.shape
        
        # Get memberships and centers using the fuzzy c-means clustering
        memberships, centers = fuzzy_c_means(x, num_clusters, m)
        
        # Create hyperedge matrix to represent each hyperedge's points
        # Initialized with -1s for padding
        hyperedge_matrix = -torch.ones(batch_size, num_clusters, num_points, dtype=torch.long)
        for b in range(batch_size):
            for c in range(num_clusters):
                idxs = torch.where(memberships[b, :, c] > threshold)[0]
                hyperedge_matrix[b, c, :len(idxs)] = idxs
        
        # Create point to hyperedge index to indicate which hyperedges each point belongs to
        # Initialized with -1s for padding
        max_edges_per_point = (memberships > threshold).sum(dim=-1).max().item()
        point_hyperedge_index = -torch.ones(batch_size, num_points, max_edges_per_point, dtype=torch.long)
        for b in range(batch_size):
            for p in range(num_points):
                idxs = torch.where(memberships[b, p, :] > threshold)[0]
                point_hyperedge_index[b, p, :len(idxs)] = idxs
    
    # Return the three constructed tensors
    return hyperedge_matrix, point_hyperedge_index, centers
