from numpy import array
from numpy import diag
from numpy import zeros
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

# select
n_elements = 2
Sigma = Sigma[:, :n_elements]
print(f'∑ matrix with n elements selected:\n {Sigma}')

VT = VT[:n_elements, :]

print(f'3rd SVD matrix with n elements selected:\n {VT}')

# reconstruct
B = U.dot(Sigma.dot(VT))
print(f'the reconstructed matrix:\n {B}')

# transform
T = U.dot(Sigma)
print(f'first transformed B matrix:\n {T}')

T = A.dot(VT.T)

print(f'second transformed B matrix:\n {T}')
