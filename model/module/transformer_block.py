import torch.nn as nn
from model.module import Attention, Mlp
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            hidden_dim=None,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.hidden_dim = int(dim * mlp_ratio) if hidden_dim is None else hidden_dim

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=self.hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()



    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def get_complexity(self, token_len, head_rate, dim_rate,  hidden_rate):
        total_flops = 0.0
        # norm1
        total_flops += self.dim * (1 - dim_rate) * token_len

        # attn
        total_flops += self.attn.get_complexity(token_len, head_rate=head_rate, dim_rate=dim_rate)

        # norm2
        total_flops += self.dim * (1 - dim_rate) * token_len

        # mlp
        total_flops += self.mlp.get_complexity(token_len, hidden_rate=hidden_rate, dim_rate=dim_rate)
        return total_flops

