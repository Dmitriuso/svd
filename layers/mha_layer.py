import torch
import torch.nn as nn
from math import ceil
from layers.tensor_svd_compression import torch_svd_compress, torch_svd_low_rank_compress, torch_svd_reconstruct, torch_svd_lowrank_reconstruct


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.k_dim = 9
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        print(f'Q shape: {Q.shape}')
        print(f'K shape: {K.shape}')
        print(f'V shape: {V.shape}')

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        ### EXPERIMENTAL SVD COMPRESSION

        Q = torch_svd_low_rank_compress(Q, 8, "cuda")
        K = torch_svd_low_rank_compress(K, 8, "cuda")
        V = torch_svd_low_rank_compress(V, 8, "cuda")

        print(f'compressed query shape: {Q.shape}')
        print(f'compressed key shape: {K.shape}')
        print(f'compressed value shape: {V.shape}')

        # compressed_query = [batch size, query len, hid dim] ; query len = hid dim
        # compressed_key = [batch size, key len, hid dim] ; key len = hid dim
        # compressed_value = [batch size, value len, hid dim] ; value len = hid dim

        k_q_head_dim = int(ceil((Q.shape[1] * Q.shape[2]) / self.n_heads))
        k_k_head_dim = int(ceil((K.shape[1] * K.shape[2]) / self.n_heads))
        k_v_head_dim = int(ceil((V.shape[1] * V.shape[2]) / self.n_heads))
        #
        # k_head_dim = int(head_dim // 2)
        print(f'k Q head dim: {k_q_head_dim}')
        print(f'k K head dim: {k_q_head_dim}')
        print(f'k V head dim: {k_q_head_dim}')


        Q = Q.view(batch_size, -1, self.n_heads, k_q_head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, k_k_head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, k_v_head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        print(f'reshaped Q shape: {Q.shape}')
        print(f'reshaped K shape: {K.shape}')
        print(f'reshaped V shape: {V.shape}')

        print(f'permuted K shape: {K.permute(0, 1, 3, 2).shape}')

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        print(f'energy tensor before mask: {energy.shape}')

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        print(f'energy tensor after mask (energy): {energy.shape}')

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        print(f'x matrix shape after matmul: {x.shape}')

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        print(f'x matrix contiguous shape: {x.shape}')

        x = x.view(batch_size, -1, 8)

        # x = [batch size, query len, hid dim]

        print(f'x matrix shape after squeezing: {x.shape}')

        ### SVD matrix reconstruction

        x = torch_svd_lowrank_reconstruct(self.fc_q(query), 8, x, "cuda")

        # x = [batch size, query len, hid dim]

        print(f'x reconstructed tensor shape: {x.shape}')

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        print(f'mha output matrix shape: {x.shape}')

        return x, attention
