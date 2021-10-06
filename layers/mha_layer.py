import torch
import torch.nn as nn
from layers.tensor_svd_compression import torch_svd_compress


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]


        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # print(f'Q shape: {Q.shape}')
        # print(f'K shape: {K.shape}')
        # print(f'V shape: {V.shape}')

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        ### EXPERIMENTAL SVD COMPRESSION

        # compressed_q = torch_svd_compress(Q)
        # compressed_k = torch_svd_compress(K)
        # compressed_v = torch_svd_compress(V)
        #
        # print(f'compressed query shape: {compressed_q.shape}')
        # print(f'compressed key shape: {compressed_k.shape}')
        # print(f'compressed value shape: {compressed_v.shape}')

        # compressed_query = [batch size, query len, hid dim] ; query len = hid dim
        # compressed_key = [batch size, key len, hid dim] ; key len = hid dim
        # compressed_value = [batch size, value len, hid dim] ; value len = hid dim

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, query len, key len]

        # print(f'tensor before mask: {energy.shape}')

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # print(f'tensor after mask (energy): {energy.shape}')

        ### EXPERIMENTAL SVD BEFORE SOFTMAX

        # new_energy = []
        # for batch in energy:
        #     new_batch = torch_svd_compress(batch)
        #     new_energy.append(new_batch)
        #
        # energy = torch.stack(new_energy)
        #
        # print(f'compressed energy shape: {energy.shape}')

        attention = torch.softmax(energy, dim=-1)

        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x, attention