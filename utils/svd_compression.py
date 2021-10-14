import torch
from numpy import array, diag, ones, transpose, zeros
from scipy.linalg import svd
from experimental import torch_svd_low_rank_compress

# define a matrix
A = array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ]
)

# Singular-value decomposition
U, s, VT = svd(A)

# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))

# populate Sigma with n x n diagonal matrix
Sigma[: A.shape[0], : A.shape[0]] = diag(s)

# select
n_elements = 2
Sigma = Sigma[:, :n_elements]

VT = VT[:n_elements, :]

# reconstruct
B = U.dot(Sigma.dot(VT))

# transform
T = U.dot(Sigma)
T = A.dot(VT.T)


def svd_compress(A):
    U, s, VT = svd(A)
    print(f"VT first shape: {VT.shape}")
    Sigma = zeros((A.shape[0], A.shape[1]))
    # create m x n Sigma matrix
    Sigma = zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[: A.shape[0], : A.shape[0]] = diag(s)
    n_elements = int(s.shape[0])
    Sigma = Sigma[:, :n_elements]
    VT = VT[:n_elements, :]
    print(f"VT second shape: {VT.shape}")
    # reconstruct
    B = U.dot(Sigma.dot(VT))
    # transform
    T = U.dot(Sigma)
    print(f"T matrix shape: {T.shape}")
    VTT = VT.T
    print(f"VTT shape: {VTT.shape}")
    Z = A.dot(VTT)
    print(f"Z shape: {Z.shape}")
    return Z


def vital_svd_compress(A):
    U, s, VT = svd(A)
    P = transpose(A).dot(U)
    print(P.shape)
    return P


D = torch.ones((17, 256))


if __name__ == "__main__":
    print(f'shape of a vector: {D[1].shape}')
    print('*'*50)
    # print(f'vector:\n {D[0]}')
    # print('*'*50)
    # print(f'SVD compressed matrix shape: {svd_compress(D).shape}')
    # print('*'*50)
    print(f'Low rank SVD matrices shapes: {[i.shape for i in torch.svd_lowrank(D, 10)]}')
