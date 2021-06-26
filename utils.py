import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from dgl.data import CiteseerGraphDataset
from dgl.data import CoraGraphDataset


def load_data(dataset):
    if dataset == 'cora':
        data = CoraGraphDataset()
    elif dataset == 'citeseer':
        data = CiteseerGraphDataset()

    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    nxg = g.to_networkx()
    adj = nx.to_scipy_sparse_matrix(nxg, dtype=np.float)

    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, train_mask, val_mask, test_mask


def preprocess_adj(adj):
    """邻接矩阵预处理及转换元组表示。"""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def normalize_adj(adj):
    """对称规范化邻接矩阵。"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def accuracy(pred, targ):
    pred = torch.max(pred, 1)[1]
    ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    return ac


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy矩阵转换为torch张量。"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
