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


def fuzzy_c_means_gpu_optimized(x, n_clusters, m=2, epsilon=1e-3, max_iter=20):
    """
    GPU-optimized Fuzzy C-Means clustering with vectorized operations
    
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
    device = x.device

    # Initialize the membership matrix
    memberships = initialize_memberships(batch_size, num_points, n_clusters, device)
    
    # Pre-compute power for efficiency
    m_inv = 1.0 / (m - 1)
    two_over_m_minus_1 = 2.0 / (m - 1)
    
    prev_memberships = torch.zeros_like(memberships)

    for iteration in range(max_iter):
        # Vectorized cluster center update
        # memberships: (batch_size, num_points, n_clusters)
        # x: (batch_size, num_points, num_dims)
        weights = memberships ** m  # (batch_size, num_points, n_clusters)
        
        # Calculate centers using einsum for efficiency
        numerator = torch.einsum('bpc,bpd->bcd', weights, x)  # (batch_size, n_clusters, num_dims)
        denominator = weights.sum(dim=1, keepdim=True).transpose(1, 2)  # (batch_size, n_clusters, 1)
        centers = (numerator / denominator).transpose(1, 2)  # (batch_size, num_dims, n_clusters)
        
        # Vectorized membership update
        # Compute all pairwise distances at once
        # x: (batch_size, num_points, num_dims) -> (batch_size, num_points, 1, num_dims)
        # centers: (batch_size, num_dims, n_clusters) -> (batch_size, 1, n_clusters, num_dims)
        x_expanded = x.unsqueeze(2)  # (batch_size, num_points, 1, num_dims)
        centers_expanded = centers.transpose(1, 2).unsqueeze(1)  # (batch_size, 1, n_clusters, num_dims)
        
        # Compute squared Euclidean distances efficiently
        diff = x_expanded - centers_expanded  # (batch_size, num_points, n_clusters, num_dims)
        distances_sq = (diff ** 2).sum(dim=-1)  # (batch_size, num_points, n_clusters)
        
        # Avoid division by zero
        distances_sq = torch.clamp(distances_sq, min=1e-10)
        
        # Compute membership using vectorized operations
        distances_powered = distances_sq ** m_inv  # (batch_size, num_points, n_clusters)
        
        # For each point and cluster, compute 1 / sum of (d_ik / d_ij)^(2/(m-1))
        # distances_powered: (batch_size, num_points, n_clusters)
        ratio_matrix = distances_powered.unsqueeze(3) / distances_powered.unsqueeze(2)  # (batch_size, num_points, n_clusters, n_clusters)
        denominator_memberships = ratio_matrix.sum(dim=3)  # (batch_size, num_points, n_clusters)
        
        memberships = 1.0 / denominator_memberships
        
        # Handle numerical issues
        memberships = torch.nan_to_num(memberships, nan=1.0/n_clusters, posinf=1.0, neginf=0.0)
        
        # Normalize memberships to ensure they sum to 1 for each point
        memberships_sum = memberships.sum(dim=2, keepdim=True)
        memberships = memberships / torch.clamp(memberships_sum, min=1e-10)

        # Check convergence using L2 norm
        if iteration > 0:
            diff_norm = torch.norm(prev_memberships - memberships)
            if diff_norm < epsilon:
                break
        prev_memberships = memberships.clone()

    return memberships, centers


def fuzzy_c_means_ultra_fast(x, n_clusters, m=2, epsilon=1e-3, max_iter=15):
    """
    Ultra-fast GPU-optimized Fuzzy C-Means with further optimizations
    
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
    device = x.device

    # Initialize centers randomly for better speed
    random_indices = torch.randint(0, num_points, (batch_size, n_clusters), device=device)
    centers = torch.zeros(batch_size, num_dims, n_clusters, device=device)
    for b in range(batch_size):
        centers[b, :, :] = x[b, random_indices[b], :].t()
    
    # Pre-compute constants
    m_inv = 1.0 / (m - 1)
    two_over_m_minus_1 = 2.0 / (m - 1)
    
    # Use mixed precision for speed (if available)
    use_mixed_precision = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    
    if use_mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            return _fuzzy_c_means_core(x, centers, n_clusters, m, m_inv, epsilon, max_iter)
    else:
        return _fuzzy_c_means_core(x, centers, n_clusters, m, m_inv, epsilon, max_iter)


def _fuzzy_c_means_core(x, centers, n_clusters, m, m_inv, epsilon, max_iter):
    """Core FCM computation with optimized tensor operations"""
    batch_size, num_points, num_dims = x.shape
    device = x.device
    
    # Initialize memberships
    memberships = torch.rand(batch_size, num_points, n_clusters, device=device)
    memberships = memberships / memberships.sum(dim=2, keepdim=True)
    
    prev_memberships = torch.zeros_like(memberships)
    
    for iteration in range(max_iter):
        # Update centers using optimized einsum
        weights = memberships ** m
        numerator = torch.einsum('bpc,bpd->bcd', weights, x)  # (batch_size, n_clusters, num_dims)
        denominator = weights.sum(dim=1, keepdim=True).transpose(1, 2)  # (batch_size, n_clusters, 1)
        centers = (numerator / denominator).transpose(1, 2)  # (batch_size, num_dims, n_clusters)
        
        # Compute distances using batch matrix multiplication for speed
        # x: (batch_size, num_points, num_dims)
        # centers: (batch_size, num_dims, n_clusters)
        x_norm_sq = (x ** 2).sum(dim=2, keepdim=True)  # (batch_size, num_points, 1)
        centers_norm_sq = (centers ** 2).sum(dim=1, keepdim=True)  # (batch_size, 1, n_clusters)
        cross_term = torch.bmm(x, centers)  # (batch_size, num_points, n_clusters)
        
        distances_sq = x_norm_sq + centers_norm_sq - 2 * cross_term
        distances_sq = torch.clamp(distances_sq, min=1e-10)
        
        # Efficient membership update
        distances_powered = distances_sq ** (-m_inv)
        memberships = distances_powered / distances_powered.sum(dim=2, keepdim=True)
        
        # Handle edge cases
        memberships = torch.nan_to_num(memberships, nan=1.0/n_clusters)
        
        # Early stopping check
        if iteration > 0 and torch.norm(prev_memberships - memberships) < epsilon:
            break
        prev_memberships = memberships.clone()
    
    return memberships, centers


# Keep original function for backward compatibility
fuzzy_c_means = fuzzy_c_means_gpu_optimized


def construct_hyperedges_optimized(x, num_clusters, threshold=0.01, m=2, use_ultra_fast=True):
    """
    GPU-optimized hyperedge construction with vectorized operations.

    Args:
        x (torch.Tensor): Input point cloud data with shape (batch_size, num_dims, num_points, 1).
        num_clusters (int): Number of clusters (hyperedges).
        threshold (float): Threshold value for memberships to consider a point belonging to a cluster.
        m (float): Fuzzifier for fuzzy c-means clustering.
        use_ultra_fast (bool): Whether to use the ultra-fast implementation.

    Returns:
        hyperedge_matrix (torch.Tensor): Tensor of shape (batch_size, n_clusters, num_points_index).
        point_hyperedge_index (torch.Tensor): Tensor of shape (batch_size, num_points, cluster_index).
        hyperedge_features (torch.Tensor): Tensor of shape (batch_size, num_dims, n_clusters).
    """
    
    with torch.no_grad():
        x = x.detach()
        batch_size, num_dims, num_points, _ = x.shape
        device = x.device
        
        # Choose FCM implementation
        if use_ultra_fast:
            memberships, centers = fuzzy_c_means_ultra_fast(x, num_clusters, m)
        else:
            memberships, centers = fuzzy_c_means_gpu_optimized(x, num_clusters, m)
        
        # Vectorized threshold application
        membership_mask = memberships > threshold  # (batch_size, num_points, n_clusters)
        
        # Handle case where no clear clustering emerges
        max_membership = memberships.max(dim=2, keepdim=True)[0]
        weak_clustering_mask = max_membership < 0.1
        
        if weak_clustering_mask.any():
            # For weak clustering, assign each point to its best cluster
            best_clusters = memberships.argmax(dim=2)  # (batch_size, num_points)
            binary_memberships = torch.zeros_like(memberships)
            batch_indices = torch.arange(batch_size, device=device)[:, None]
            point_indices = torch.arange(num_points, device=device)[None, :]
            binary_memberships[batch_indices, point_indices, best_clusters] = 1.0
            
            # Apply only where clustering is weak
            memberships = torch.where(weak_clustering_mask, binary_memberships, memberships)
            membership_mask = memberships > 0.5
        
        # Efficient hyperedge matrix construction
        # Count points per hyperedge
        points_per_edge = membership_mask.sum(dim=1)  # (batch_size, n_clusters)
        max_points_per_edge = points_per_edge.max().item()
        
        if max_points_per_edge == 0:
            max_points_per_edge = 1
        
        hyperedge_matrix = -torch.ones(
            batch_size, num_clusters, max_points_per_edge,
            dtype=torch.long, device=device
        )
        
        # Vectorized assignment using advanced indexing
        for b in range(batch_size):
            for c in range(num_clusters):
                point_indices = torch.where(membership_mask[b, :, c])[0]
                if len(point_indices) > 0:
                    hyperedge_matrix[b, c, :len(point_indices)] = point_indices
        
        # Efficient point to hyperedge index construction
        edges_per_point = membership_mask.sum(dim=2)  # (batch_size, num_points)
        max_edges_per_point = edges_per_point.max().item()
        
        if max_edges_per_point == 0:
            max_edges_per_point = 1
        
        point_hyperedge_index = -torch.ones(
            batch_size, num_points, max_edges_per_point,
            dtype=torch.long, device=device
        )
        
        for b in range(batch_size):
            for p in range(num_points):
                edge_indices = torch.where(membership_mask[b, p, :])[0]
                if len(edge_indices) > 0:
                    point_hyperedge_index[b, p, :len(edge_indices)] = edge_indices
                else:
                    # Assign to best cluster if no assignment
                    best_cluster = memberships[b, p, :].argmax()
                    point_hyperedge_index[b, p, 0] = best_cluster
    
    return hyperedge_matrix, point_hyperedge_index, centers


# Update the main function to use optimized version
construct_hyperedges = construct_hyperedges_optimized
