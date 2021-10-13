import torch

A = torch.zeros((128, 18, 6))
B = A.view(128, -1, 8, 9).permute(0, 2, 1, 3)
# C = A.reshape(A, [128, -1, 8, 256]).permute(0, 2, 1, 3)

a_elements = A.shape[0]*A.shape[1]*A.shape[2]

if __name__ == '__main__':
    print(f'A shape: {A.shape}')
    print(f'A n_elements: {a_elements}')
    print(B.shape)
    # print(C.shape)
