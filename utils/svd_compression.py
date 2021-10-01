import torch
from numpy import array, ones, zeros, diag, transpose
from scipy.linalg import svd

# define a matrix
A = array([
    [1,2,3,4,5,6,7,8,9,10],
    [11,12,13,14,15,16,17,18,19,20],
    [21,22,23,24,25,26,27,28,29,30]])
print(f'the initial matrix A:\n {A}')

# Singular-value decomposition
U, s, VT = svd(A)

print(f'SVD of the matrix A:\n {U.shape} \n{"*"*50}\n {s} \n{"*"*50}\n {VT}')
print('*'*150)

# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))

# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = diag(s)

print(f'∑ matrix populated with diag matrix after SVD:\n {Sigma}')
print('*'*150)

# select
n_elements = 2
Sigma = Sigma[:, :n_elements]
print(f'∑ matrix with n elements selected:\n {Sigma}')
print('*'*150)

VT = VT[:n_elements, :]

print(f'3rd SVD matrix with n elements selected:\n {VT}')
print('*'*150)

# reconstruct
B = U.dot(Sigma.dot(VT))
print(f'the reconstructed matrix:\n {B}')

# transform
T = U.dot(Sigma)
print(f'first transformed B matrix:\n {T}')

T = A.dot(VT.T)

print(f'second transformed B matrix:\n {T}')


def svd_compress(A):
    U, s, VT = svd(A)
    Sigma = zeros((A.shape[0], A.shape[1]))
    # create m x n Sigma matrix
    Sigma = zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[:A.shape[0], :A.shape[0]] = diag(s)
    n_elements = int(s.shape[0])
    Sigma = Sigma[:, :n_elements]
    VT = VT[:n_elements, :]
    # reconstruct
    B = U.dot(Sigma.dot(VT))
    # transform
    T = U.dot(Sigma)
    T = A.dot(VT.T)
    return T

def vital_svd_compress(A):
    U, s, VT = svd(A)
    P = transpose(A).dot(U)
    print(P.shape)
    return P


D = ones((128, 17, 256))


if __name__ == '__main__':
    print('*'*50)
    print(f'checking the SVD compression function:\n {svd_compress(D).shape}')