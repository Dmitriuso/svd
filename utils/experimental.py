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
M = zeros(12, 256)
C = zeros(128, 60, 256)

A = tensor(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ],
    dtype=torch.float32,
)

# permuted = M.permute(0, 2, 1)


svdlr = svd_lowrank(M, q=6)

# for i in A:
#     print(i.shape)
#     B = i.cpu().data.numpy()
#     compressed = svd_compress(B)
#     print(compressed.shape)


def torch_svd_compress(A):
    print(f"def input shape: {A.shape}")
    U, s, VT = svd(A)
    print(f"U matrix shape: {U.shape}")
    print(f"s matrix shape: {s.shape}")
    print(f"VT first shape: {VT.shape}")
    # create m x n Sigma matrix
    Sigma = zeros((A.shape[0], A.shape[1]))
    print(f"Sigma matrix first shape: {Sigma.shape}")
    # populate Sigma with n x n diagonal matrix
    Sigma[: A.shape[0], : A.shape[0]] = diag(s)
    n_elements = int(s.shape[0])
    Sigma = Sigma[:, :n_elements]
    print(f"Sigma matrix second shape: {Sigma.shape}")
    VT = VT[:n_elements, :]
    print(f"VT second shape: {VT.shape}")
    # reconstruct
    # C = mm(Sigma, VT)
    # B = mm(U, C)
    # transform
    # T = mm(U, Sigma)
    # print(f'T matrix shape: {T.shape}')
    VTT = transpose(VT, 0, 1)
    print(f"VTT shape: {VTT.shape}")
    Z = mm(A, VTT)
    return Z


def torch_svd_low_rank_compress(A, q):
    print(f"def input shape: {A.shape}")
    U, s, VT = svd_lowrank(A, q=q)
    print(f"U matrix shape: {U.shape}")
    print(f"s matrix shape: {s.shape}")
    print(f"VT first shape: {VT.shape}")
    # create m x n Sigma matrix
    Sigma = zeros((q, q))
    print(f"Sigma matrix first shape: {Sigma.shape}")
    # populate Sigma with n x n diagonal matrix
    Sigma[: q, : q] = diag(s)
    n_elements = int(q)
    Sigma = Sigma[:, :n_elements]
    print(f"Sigma matrix second shape: {Sigma.shape}")
    VT2 = VT[:n_elements, :]
    print(f"VT second shape: {VT.shape}")
    # reconstruct
    # C = mm(Sigma, VT)
    # B = mm(U, C)
    # transform
    T = mm(U, Sigma)
    print(f"T matrix shape: {T.shape}")
    VTT = transpose(VT2, 0, 1)
    print(f"VTT shape: {VTT.shape}")
    Z = mm(T, VTT)
    print(f'torch low rank compressed matrix Z: {Z.shape}')
    return Z


def torch_svd_low_rank_compress_3d(A, q):
    new_tensors_list = []
    for i in A:
        U, s, VT = svd_lowrank(i, q=q)
        # create m x n Sigma matrix
        Sigma = zeros((q, q))
        # populate Sigma with n x n diagonal matrix
        Sigma[: q, : q] = diag(s)
        n_elements = int(q)
        Sigma = Sigma[:, :n_elements]
        VT = VT[:n_elements, :]
        # transform
        T = mm(U, Sigma)
        VTT = transpose(VT, 0, 1)
        Z = mm(T, VTT)
        print(f'Z shape: {Z.shape}')
        new_tensors_list.append(Z)
    K = stack(new_tensors_list)
    return K


def torch_svd_reconstruct(A, B):
    U, s, VT = svd(A)
    # create m x n Sigma matrix
    Sigma = zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[: A.shape[0], : A.shape[0]] = diag(s)
    n_elements = int(s.shape[0])
    Sigma = Sigma[:, :n_elements]
    VT = VT[:n_elements, :]
    # reconstruct
    C = mm(Sigma, VT)
    B = mm(U, C)
    return B


tensor_list = []

# for i in permuted:
#     print(f'i shape: {i.shape}')
#     print(f'i type: {type(i)}')
#     # B = tensor(i)
#     torch_compressed = torch_svd_compress(i)
#     print(f'shape of torch-compressed matrix: {torch_compressed.shape}')
#     print(f'type of torch-compressed matrix: {type(torch_compressed)}')
#     tensor_list.append(torch_compressed)


# K = stack(tensor_list)
# print(f'shape of torch-compressed tensor: {K.shape}')


if __name__ == "__main__":
    # print(f"A matrix shape:\n {M.shape}")
    # print("*" * 50)
    # print(f'permuted matrix shape:\n {permuted.shape}')
    # print('*'*50)
    # print(f'the SVD matrices:\nU: {U.shape}\ns:{s.shape}\nVT: {VT.shape}')
    # print('*'*50)
    # print(f'∑ matrix populated with diag matrix after SVD:\n {Sigma.shape}')
    # print('*'*50)
    print(f'the shape of the torch compressed lowrank  matrix M: {torch_svd_low_rank_compress(M, 10).shape}')
    print("*" * 50)
    # print(f"the torch compressed matrix M:\n {torch_svd_compress(A).shape}")
    # print("*" * 50)
    # print(f"the reconstructed after compression matrix A:\n {torch_svd_reconstruct(A)}")
    # print("*"*50)
    # print(f'the shape of 3D tensor after low rank compression and stack back: {torch_svd_low_rank_compress_3d(C, 50).shape}')
