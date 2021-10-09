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
