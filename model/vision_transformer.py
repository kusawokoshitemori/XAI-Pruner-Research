from .module import *
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import  trunc_normal_
import torch.nn as nn
import torch

class VisionTransformer(VisionTransformer):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            act_layer=None,
            pruned=False,
            config = None
    ):
        super().__init__()
        act_layer = act_layer or nn.GELU
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.img_size = img_size
        self.depth = depth

        if not pruned:
            self.embed_dim = embed_dim  # num_features for consistency with other models

            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=self.embed_dim)

            self.num_patches = self.patch_embed.num_patches

            self.mlp_ratio = mlp_ratio
            self.num_heads = num_heads
            self.hidden_dim = int(embed_dim * mlp_ratio)

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.blocks = nn.Sequential(*[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                )
                for i in range(depth)])

            self.norm = norm_layer(embed_dim)

            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.apply(self._init_weights)

        else:
            assert config is not None, "Pruned config is not provided."

            self.embed_dim = config["embed_dim"]
            self.depth = len(config["num_heads"])

            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=self.embed_dim)

            self.num_patches = self.patch_embed.num_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim))

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

            self.blocks = nn.ModuleList()
            for i in range(self.depth):
                num_heads = config["num_heads"][i]
                hidden_dim = config["hidden_dim"][i]

                self.blocks.append(Block(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=None,
                    hidden_dim = hidden_dim,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer))

            self.norm = norm_layer(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_complexity(self,dim_rate=0.0, hidden_rate=0.0, head_rate=0.0):
        total_flops = 0.0

        # patch_embed
        total_flops += self.patch_embed.get_complexity(dim_rate=dim_rate)

        # blocks
        for blk in self.blocks:
            total_flops += blk.get_complexity(token_len=self.num_patches+1, dim_rate=dim_rate, hidden_rate=hidden_rate, head_rate=head_rate)

        # head
        total_flops += self.embed_dim * ( 1 - dim_rate) * self.num_classes
        return total_flops




