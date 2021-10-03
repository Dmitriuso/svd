from torch import ones, zeros, diag, svd_lowrank, mm, transpose, tensor, stack
from torch.linalg import svd


class SVD_compress:
    def __init__(self, tensor_3d):
        self.A = tensor_3d

    @classmethod
    def torch_svd_compress(self, matrix):
        U, s, VT = svd(matrix)
        Sigma = zeros((matrix.shape[0], matrix.shape[1]))
        # create m x n Sigma matrix
        Sigma = zeros((matrix.shape[0], matrix.shape[1]))
        # populate Sigma with n x n diagonal matrix
        Sigma[:matrix.shape[0], :matrix.shape[0]] = diag(s)
        n_elements = int(s.shape[0])
        Sigma = Sigma[:, :n_elements]
        VT = VT[:n_elements, :]
        # transform
        T = mm(U, Sigma)
        VTT = transpose(VT, 0, 1)
        Z = mm(matrix, VTT)
        return Z

    def iter_and_stack(self):
        new_tensors_list = []
        for i in self.A:
            torch_compressed = self.torch_svd_compress(i)
            new_tensors_list.append(torch_compressed)
        K = stack(new_tensors_list)
        return K
