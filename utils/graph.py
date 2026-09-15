import numpy as np
import torch

def construct_H_with_KNN_from_distance(dis_mat, k_neig=9):
    dis_mat = dis_mat.copy()
    n_obj = dis_mat.shape[0]
    H = np.zeros((n_obj, n_obj))
    for i in range(n_obj):
        dis_mat[i, i] = 0
        idx = np.argsort(dis_mat[i])[:k_neig]
        H[i, idx] = 1.0
    return H

def generate_G_from_H(H):
    H = np.array(H)
    W = np.ones(H.shape[1])
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE = np.diag(1.0 / (DE + 1e-8))
    DV2 = np.diag(1.0 / np.sqrt(DV + 1e-8))
    HT = H.T
    G = DV2 @ H @ invDE @ HT @ DV2
    return torch.tensor(G, dtype=torch.float32)