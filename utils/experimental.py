from torch import ones, zeros, diag
from torch.linalg import svd
import torch.nn as nn

# define a matrix
A = ones(128, 17, 256)
print(A.shape)

U, s, VT = svd(A)

# create m x n Sigma matrix
Sigma = zeros(A.shape[0], A.shape[1], A.shape[2])

# populate Sigma with n x n diagonal matrix
# Sigma[:A.shape[1], :A.shape[1]] = diag(s)

if __name__ == '__main__':
    print(f'the input matrix:\n {A.shape}')
    print('*'*50)
    print(f'the SVD matrices:\nU: {U.shape}\ns:{s.shape}\nVT: {VT.shape}')
    print('*'*50)
    print(f'∑ matrix populated with diag matrix after SVD:\n {Sigma.shape}')



