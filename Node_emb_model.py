import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


def degree(index, num_nodes, dtype=None, device=None):
    """计算稀疏图中每个节点的度（与 torch_geometric.utils.degree 兼容）"""
    if dtype is None:
        dtype = torch.float32
    deg = torch.zeros(num_nodes, dtype=dtype, device=device)
    deg.scatter_add_(0, index, torch.ones_like(index, dtype=dtype))
    return deg# ------------------------------------------------------------
# 核心模块：不拆分的双邻接矩阵 GCN
# ------------------------------------------------------------
class node_embedding(nn.Module):
    """
    两层 GCN，使用两个不同的邻接矩阵分别聚合，输入特征不拆分。
    每层：
        agg = α·(A_ot @ X) + (1-α)·(A_t @ X)
        X = LeakyReLU(W·agg)
    """
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, alpha=0.6, dropout=0.5):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        if out_dim is None:
            out_dim = in_dim
        self.alpha = alpha
        self.dropout = dropout

        self.W1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.leaky_relu = nn.LeakyReLU()

    def normalize_adj(self, adj):
        """
        对称归一化：A_hat = D^{-1/data} (A+I) D^{-1/data}
        支持 torch.Tensor (dense/sparse) 和 scipy.sparse 矩阵
        """
        if isinstance(adj, torch.Tensor):
            if adj.is_sparse:
                adj = adj.coalesce()
                indices = adj.indices()
                values = adj.values()
                n = adj.size(0)
                device = adj.device

                # 添加自环
                self_loop_indices = torch.arange(n, device=device)
                self_loop_indices = torch.stack([self_loop_indices, self_loop_indices])
                indices = torch.cat([indices, self_loop_indices], dim=1)
                values = torch.cat([values, torch.ones(n, device=device)])

                # 度矩阵
                deg = degree(indices[0], n, dtype=torch.float, device=device)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

                row, col = indices
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                values = values * norm

                return torch.sparse_coo_tensor(indices, values, (n, n), device=device)
            else:
                # 稠密矩阵
                n = adj.size(0)
                device = adj.device
                adj = adj + torch.eye(n, device=device)
                deg = adj.sum(dim=1)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                D_inv_sqrt = torch.diag(deg_inv_sqrt)
                return D_inv_sqrt @ adj @ D_inv_sqrt
        else:
            # scipy.sparse 矩阵
            adj = adj + sp.eye(adj.shape[0])
            deg = np.array(adj.sum(axis=1)).flatten()
            deg_inv_sqrt = np.power(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0
            D_inv_sqrt = sp.diags(deg_inv_sqrt)
            adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
            # 转换为 torch sparse tensor
            adj_coo = adj_norm.tocoo()
            indices = torch.LongTensor([adj_coo.row, adj_coo.col])
            values = torch.FloatTensor(adj_coo.data)
            return torch.sparse_coo_tensor(indices, values, adj.shape)

    def forward(self, adj_t, adj_ot, X_t, X_ot):
        adj_t_norm = self.normalize_adj(adj_t)
        print(type(X_t))
        X_t = torch.tensor(X_t, dtype=torch.float32)
        X_ot = torch.tensor(X_ot, dtype=torch.float32)

        if adj_t_norm.is_sparse:
            agg_t = torch.sparse.mm(adj_t_norm, X_t)
        else:
            agg_t = adj_t_norm @ X_t

        if adj_ot.is_sparse:
            agg_ot = torch.sparse.mm(adj_ot, X_ot)
        else:
            agg_ot = adj_ot @ X_ot

        X_t1 = self.leaky_relu(self.W1(agg_t))
        X_t1 = F.dropout(X_t1, self.dropout, training=self.training)

        X_ot1 = self.leaky_relu(self.W1(agg_ot))
        X_ot1 = F.dropout(X_ot1, self.dropout, training=self.training)

        if adj_t_norm.is_sparse:
            agg_t2 = torch.sparse.mm(adj_t_norm, X_t1)
            agg_ot2 = torch.sparse.mm(adj_t_norm, X_ot1)
        else:
            agg_t2 = adj_t_norm @ X_t1
            agg_ot2 = adj_t_norm @ X_ot1

        X_t2 = self.leaky_relu(self.W2(agg_t2))
        X_t2 = F.dropout(X_t2, self.dropout, training=self.training)

        X_ot2 = self.leaky_relu(self.W2(agg_ot2))
        X_ot2 = F.dropout(X_ot2, self.dropout, training=self.training)

        combined = self.alpha * X_ot2 + (1 - self.alpha) * X_t2

        return combined