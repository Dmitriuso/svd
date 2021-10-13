import torch
from torch import (
    diag,
    dot,
    mm,
    ones,
    permute,
    stack,
    svd_lowrank,
    tensor,
    transpose,
    zeros,
)
from torch.linalg import svd

# define a matrix
A = zeros(12, 256)
O = zeros(12, 12)
N = zeros(128, 12, 256)

G = N[1]

low_rank = 6

new_tensors_list = []

for i in N:
    U, s, VT = svd(i)
    print(f"U matrix shape: {U.shape}")
    print(f"s matrix shape: {s.shape}")
    print(f"VT first shape: {VT.shape}")

    # create m x n Sigma matrix
    Sigma = zeros((i.shape[0], i.shape[1]))
    print(f"Sigma matrix first shape: {Sigma.shape}")

    # populate Sigma with n x n diagonal matrix
    Sigma[: i.shape[0], : i.shape[0]] = diag(s)
    print(f"Sigma matrix second shape: {Sigma.shape}")

    n_elements = int(s.shape[0])
    Sigma = Sigma[:, :n_elements]
    print(f"Sigma matrix second shape: {Sigma.shape}")

    VT = VT[:n_elements, :]
    print(f"VT second shape: {VT.shape}")

    # reconstruct
    C = mm(Sigma, VT)
    B = mm(O, C)
    new_tensors_list.append(B)

K = stack(new_tensors_list)

if __name__ == '__main__':
    print("*" * 50)
    print(f"the reconstructed tensor shape:\n {K.shape}")
    print("*" * 50)
    print(f'an element of a tensor: {G.shape}')
