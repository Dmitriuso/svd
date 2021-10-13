from torch import diag, mm, ones, stack, svd_lowrank, tensor, transpose, zeros
from torch.linalg import svd


def torch_svd_compress(matrix):
    new_tensors_list = []
    for i in matrix:
        U, s, VT = svd(i)
        # create m x n Sigma matrix
        Sigma = zeros((i.shape[0], i.shape[1]))
        # populate Sigma with n x n diagonal matrix
        Sigma[: i.shape[0], : i.shape[0]] = diag(s)
        n_elements = int(s.shape[0])
        Sigma = Sigma[:, :n_elements]
        VT = VT[:n_elements, :]
        # transform
        VTT = transpose(VT, 0, 1)
        Z = mm(i, VTT)
        new_tensors_list.append(Z)
    K = stack(new_tensors_list)
    return K

def torch_svd_low_rank_compress(A, q, device):
    new_tensors_list = []
    for i in A:
        U, s, VT = svd_lowrank(i, q=q)
        # create m x n Sigma matrix
        Sigma = zeros((q, q)).to(device)
        # populate Sigma with n x n diagonal matrix
        Sigma[: q, : q] = diag(s)
        n_elements = int(q)
        Sigma = Sigma[:, :n_elements]
        VT = VT[:n_elements, :]
        # transform
        T = mm(U, Sigma)
        VTT = transpose(VT, 0, 1)
        Z = mm(T, VTT)
        new_tensors_list.append(Z)
    K = stack(new_tensors_list)
    return K


def torch_svd_reconstruct(N, O, device):
    new_tensors_list = []

    for i in range(N.shape[0]):
        Q = N[i]
        U, s, VT = svd(Q)
        # create m x n Sigma matrix
        Sigma = zeros((Q.shape[0], Q.shape[1])).to(device)

        # populate Sigma with n x n diagonal matrix
        Sigma[: Q.shape[0], : Q.shape[0]] = diag(s)

        n_elements = int(s.shape[0])
        Sigma = Sigma[:, :n_elements]
        VT = VT[:n_elements, :]

        # reconstruct
        C = mm(Sigma, VT)
        B = mm(O[i], C)
        new_tensors_list.append(B)

    K = stack(new_tensors_list)
    return K
