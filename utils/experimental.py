from torch import ones, zeros, diag, svd_lowrank
from torch.linalg import svd
import torch.nn as nn
from svd_compression import svd_compress

# define a matrix
A = zeros(128, 17, 256)
print(A.shape)

U, s, VT = svd_lowrank(A, q=17)

# create m x n Sigma matrix
Sigma = zeros(A.shape[0], A.shape[1])

# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = diag(s)

svdlr = svd_lowrank(A, q=2)

for i in A:
    print(i.shape)
    B = i.cpu().data.numpy()
    compressed = svd_compress(B)
    print(compressed.shape)


if __name__ == '__main__':
    print(f'the input matrix:\n {A.shape}')
    print('*'*50)
    print(f'the SVD matrices:\nU: {U.shape}\ns:{s.shape}\nVT: {VT.shape}')
    print('*'*50)
    print(f'∑ matrix populated with diag matrix after SVD:\n {Sigma.shape}')
    print('*'*50)
    print(f'the SVD of a low-rank matrix A:\n {[i.shape for i in svd_lowrank(A)]}')

