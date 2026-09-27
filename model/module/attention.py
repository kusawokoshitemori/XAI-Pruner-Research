import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        head_dim = 64
        self.head_dim = head_dim

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_complexity(self, token_len, head_rate, dim_rate):
        total_flops = 0.0
        # qkv
        total_flops += token_len * self.dim * (1 - dim_rate)  * self.head_dim * self.num_heads * (1 - head_rate) * 3

        # qkv attention
        total_flops += 2 * token_len * token_len * self.head_dim * self.num_heads * (1 - head_rate)

        # proj
        total_flops += token_len * self.head_dim * self.num_heads * (1 - head_rate) * self.dim * (1 - dim_rate)
        return total_flops


