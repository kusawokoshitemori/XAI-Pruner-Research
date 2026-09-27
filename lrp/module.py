import torch
import torch.nn as nn
from timm.models.layers import DropPath
import inspect

import lrp.rules as rules
import torch.fx
from timm.models.resnet import BasicBlock, Bottleneck
from typing import Optional, Type


class LayerNormLRP(nn.LayerNorm):

    def __init__(self, normalized_shape, eps: float = 0.00001, elementwise_affine: bool = True, bias: bool = True, device=None, dtype=None):
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)

    def forward(self, x):
        return rules.layernorm_lrp.apply(x, self.weight, self.bias, self.eps)
        #return nd.identity_lrp.apply(super().forward, x)


class BatchNorm2dLRP(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(BatchNorm2dLRP, self).__init__(*args, **kwargs)

    def forward(self, x):
        return rules.identity_lrp.apply(super().forward, x)


class EpsilonLinearLRP(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, epsilon=1e-6, **kwargs):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.epsilon = epsilon
        self.relevance = None

    def forward(self, x):
        return rules.epsilon_linear_lrp.apply(x, self.weight, self.bias, self.epsilon)


class GELULRP(nn.GELU):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return rules.identity_lrp.apply(super().forward, x)


class MlpLRP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        self.fc1 = EpsilonLinearLRP(in_features, hidden_features, bias)
        if act_layer == nn.GELU:
            self.act = GELULRP()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = EpsilonLinearLRP(hidden_features, out_features, bias)
        self.drop2 = nn.Dropout(drop)
        self.relevance = {}

    def forward(self, x):
        # x [B, N, C]
        x = self.fc1(x)
        x.register_hook(self.save_relevance("mlp-hidden"))
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def save_relevance(self, name):
        def hook(grad):
            self.relevance[name] = grad
        return hook


class AttentionLRP(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.proj_q_weight = None
        self.proj_q_bias = None
        self.proj_k_weight = None
        self.proj_k_bias = None

        self.proj_v = EpsilonLinearLRP(dim, dim,bias=False)
        self.proj = EpsilonLinearLRP(dim, dim, bias=False)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relevance = {}

    def forward(self, x):
        B, N, C = x.shape
        device = x.device
        self.proj_q_weight = self.proj_q_weight.to(device)
        self.proj_k_weight = self.proj_k_weight.to(device)

        if self.proj_q_bias is not None:
            self.proj_q_bias = self.proj_q_bias.to(device)
            self.proj_k_bias = self.proj_k_bias.to(device)

        with torch.no_grad():
            q = torch.nn.functional.linear(x, self.proj_q_weight, self.proj_q_bias)
            k = torch.nn.functional.linear(x, self.proj_k_weight, self.proj_k_bias)

        v = self.proj_v(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        with torch.no_grad():
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

        x = rules.epsilon_lrp.apply(torch.matmul, 1e-6, attn.detach(), v)
        x.register_hook(self.save_relevance('head'))
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def save_relevance(self, name):
        def hook(grad):
            self.relevance[name] = grad
        return hook


class BlockLRP(nn.Module):
    def __init__(self, top_k_percent=0.8):
        super().__init__()
        self.norm1 = None
        self.attn = None
        self.drop_path1 = None

        self.norm2 = None
        self.mlp = None
        self.drop_path2 = None

        self.filter_attn_out = Filter(top_k_percent=0)
        self.filter_mlp_out = Filter(top_k_percent=0)

        self.filter_attn_in = Filter(top_k_percent=top_k_percent)
        self.filter_mlp_in = Filter(top_k_percent=top_k_percent)


        self.relevance = {}

    def forward(self, x):
        attn = self.drop_path1(self.attn(self.norm1(self.filter_attn_out(x))))

        attn = self.filter_attn_in(attn)
        x = rules.add_tensors_lrp.apply(x, attn)

        mlp = self.drop_path2(self.mlp(self.norm2(self.filter_mlp_out(x))))

        mlp = self.filter_mlp_in(mlp)
        x = rules.add_tensors_lrp.apply(x, mlp)
        return x



class PatchEmbedLRP(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, flatten, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = None
        self.norm = None


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)

        return x

    def replace(self, original):
        self.proj = InitConv2dLRP(original.proj)



class ViTLRP(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size,
            num_classes,
            embed_dim,
            depth):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = None
        self.hidden_dim = None

        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.blocks = None
        self.norm = None

        self.head = EpsilonLinearLRP(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.relevance = {}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x.register_hook(self.save_relevance("dim"))
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def save_relevance(self, name):
        def hook(grad):
            self.relevance[name] = grad
        return hook

    def get_relevance(self):
        self.relevance["head"] = list()
        self.relevance["hidden"] = list()
        for blk in self.blocks:
            self.relevance["head"].append(blk.attn.relevance["head"])
            self.relevance["hidden"].append(blk.mlp.relevance["mlp-hidden"])

        self.relevance["dim"] = self.relevance["dim"]

        return self.relevance

    def clear_relevance(self):
        self.relevance.clear()
        self.patch_embed.relevance = None

        for blk in self.blocks:
           blk.attn.relevance.clear()
           blk.mlp.relevance.clear()


class Filter(nn.Module):
    def __init__(self, top_k_percent=0.5):
        super().__init__()
        self.top_k = top_k_percent

    def forward(self, x):
        return rules.filter_lrp.apply(x, self.top_k)


class ReLULRP(nn.ReLU):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return rules.identity_lrp.apply(nn.ReLU(), x)


class Conv2dLRP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dLRP, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.relevance = None
        self.filter = Filter(top_k_percent=0.5)

    def forward(self, x):
        outputs = rules.epsilon_conv2d_lrp.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
        outputs.register_hook(self.save_relevance())
        outputs = self.filter(outputs)

        return outputs

    def save_relevance(self):
        def hook(grad):
            self.relevance = grad
        return hook

    def get_relevance(self):
        return torch.mean(torch.sum(torch.abs(self.relevance), dim=(2, 3)), dim=0).cpu().numpy()

    def clear_relevance(self):
        self.relevance = None


class AdaptiveAvgPool2dLRP(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dLRP, self).__init__(output_size=output_size)

    def forward(self, x):
        return rules.avgpool2d_lrp.apply(x, self.output_size)


class MaxPool2dLRP(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool2dLRP, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):
        return rules.maxpool2d_lrp.apply(x, self.kernel_size, self.stride, self.padding)


class BasicBlockLRP(BasicBlock):
    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None):
        super(BasicBlockLRP, self).__init__(
            inplanes,
            planes,
            stride,
            downsample,
            cardinality=cardinality,
            base_width=base_width,
            reduce_first=reduce_first,
            dilation=dilation,
            first_dilation=first_dilation,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
            aa_layer=aa_layer,
            drop_block=drop_block,
            drop_path=drop_path)

        self.residual_filter = Filter(top_k_percent=0)
        self.relevance = []

    def forward(self, x):
        residual = x
        x = self.residual_filter(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x = rules.add_tensors_lrp.apply(x, residual)
        x = self.act2(x)
        return x

    def get_relevance(self):
        relevance = list()
        relevance.append(self.conv1.get_relevance())
        relevance.append(self.conv2.get_relevance())
        return relevance

    def clear_relevance(self):
        self.conv1.clear_relevance()
        self.conv2.clear_relevance()
        if self.downsample is not None:
            self.downsample[0].clear_relevance()


class BottleneckLRP(Bottleneck):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None
    ):
        super(BottleneckLRP, self).__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            cardinality=cardinality,
            base_width=base_width,
            reduce_first=reduce_first,
            dilation=dilation,
            first_dilation=first_dilation,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
            aa_layer=aa_layer,
            drop_block=drop_block,
            drop_path=drop_path
        )
        self.residual_filter = Filter(top_k_percent=0)
        self.relevance = []

    def forward(self, x):
        shortcut = x

        x = self.residual_filter(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = rules.add_tensors_lrp.apply(x, shortcut)
        x = self.act3(x)

        return x

    def get_relevance(self):
        relevance = list()
        relevance.append(self.conv1.get_relevance())
        relevance.append(self.conv2.get_relevance())
        relevance.append(self.conv3.get_relevance())
        return relevance


def InitReLULRP(original):
    replacement = ReLULRP()
    return replacement


def InitConv2dLRP(original):
    replacement = Conv2dLRP(in_channels=original.in_channels,
                            out_channels=original.out_channels,
                            kernel_size=original.kernel_size,
                            stride=original.stride,
                            padding=original.padding)

    replacement.weight = original.weight
    if original.bias is not None:
        replacement.bias = original.bias

    return replacement


def InitAdaptiveAvgPool2dLRP(original):
    replacement = AdaptiveAvgPool2dLRP(output_size=original.output_size)
    return replacement


def InitMaxPool2dLRP(original):
    kwargs = {}
    for arg in inspect.signature(original.__init__).parameters.keys():
        if hasattr(original, arg):
            kwargs[arg] = getattr(original, arg)

    replacement = MaxPool2dLRP(**kwargs)
    return replacement


def InitBasicBlockLRP(original):

   inplanes= original.conv1.in_channels
   planes = int(original.conv2.out_channels / original.expansion)
   stride = original.stride
   downsample = original.downsample

   replacement = BasicBlockLRP(inplanes, planes, stride, downsample)

   replacement.load_state_dict(original.state_dict(), strict=False)
   return replacement


def InitBottleneckLRP(original):
    inplanes= original.conv1.in_channels
    planes = int(original.conv3.out_channels / original.expansion)
    stride = original.stride
    downsample = original.downsample

    replacement = BottleneckLRP(inplanes, planes, stride, downsample)
    replacement.load_state_dict(original.state_dict(), strict=False)
    return replacement


def InitLayerNormLRP(original):
    kwargs = {}
    for arg in inspect.signature(original.__init__).parameters.keys():
        if hasattr(original, arg):
            kwargs[arg] = getattr(original, arg)

    kwargs["bias"] = True if original.bias is not None else False

    replacement = LayerNormLRP(**kwargs)
    replacement.load_state_dict(original.state_dict())

    return replacement


def InitAttentionLRP(original):
    dim = original.qkv.weight.shape[1]
    attn_drop = original.attn_drop.p
    proj_drop = original.proj_drop.p

    replacement = AttentionLRP(dim=dim, num_heads=original.num_heads, attn_drop=attn_drop, proj_drop=proj_drop)


    replacement.proj_q_weight = original.qkv.weight[:dim]
    replacement.proj_k_weight = original.qkv.weight[dim:dim*2]
    replacement.proj_v.weight = nn.Parameter(original.qkv.weight[dim*2:dim*3])

    if original.qkv.bias is not None:
        replacement.proj_q_bias = original.qkv.bias[:dim]
        replacement.proj_k_bias= original.qkv.bias[dim:dim * 2]
        replacement.proj_v.bias = nn.Parameter(original.qkv.bias[dim * 2:dim * 3])


    replacement.proj.weight = original.proj.weight
    if original.proj.bias is not None:
        replacement.proj.bias = original.proj.bias

    return replacement


def InitMlpLRP(original):
    in_features = original.in_features
    hidden_features = original.hidden_features
    out_features = original.out_features

    drop = original.drop1.p

    replacement = MlpLRP(in_features, hidden_features, out_features, act_layer=nn.GELU, bias=True, drop=drop)

    replacement.fc1.weight = original.fc1.weight
    if original.fc1.bias is not None:
        replacement.fc1.bias = original.fc1.bias

    replacement.fc2.weight = original.fc2.weight
    if original.fc2.bias is not None:
        replacement.fc2.bias = original.fc2.bias


    return replacement


def InitViTBlockLRP(original, top_k_percent=0.5):
    replacement = BlockLRP(top_k_percent=top_k_percent)

    drop_path = original.drop_path1.p if isinstance(original.drop_path1, DropPath) else 0

    replacement.norm1 = InitLayerNormLRP(original.norm1)
    replacement.attn = InitAttentionLRP(original.attn)
    replacement.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    replacement.norm2 = InitLayerNormLRP(original.norm2)
    replacement.mlp = InitMlpLRP(original.mlp)
    replacement.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    return replacement


def InitPatchEmbedLRP(original):
    replacement = PatchEmbedLRP(
        img_size=original.img_size,
        patch_size=original.patch_size,
        in_chans=original.in_chans,
        embed_dim=original.embed_dim,
        flatten=original.flatten
    )

    replacement.replace(original)

    if isinstance(original.norm, nn.LayerNorm):
        replacement.norm = LayerNormLRP(replacement.embed_dim)
    else:
        replacement.norm = nn.Identity()

    return replacement


def InitViTLRP(original, top_k_percent):
    replacement = ViTLRP(
        img_size=original.img_size,
        patch_size=original.patch_size,
        num_classes=original.num_classes,
        embed_dim=original.embed_dim,
        depth=original.depth
    )

    replacement.num_heads = original.num_heads
    replacement.hidden_dim = original.hidden_dim

    replacement.patch_embed = InitPatchEmbedLRP(original.patch_embed)

    replacement.cls_token.data.copy_(original.cls_token.data)
    replacement.pos_embed.data.copy_(original.pos_embed.data)

    replacement.blocks = nn.Sequential(*[InitViTBlockLRP(original.blocks[i], top_k_percent=top_k_percent) for i in range(replacement.depth)])

    replacement.norm = InitLayerNormLRP(original.norm)
    replacement.head.weight = original.head.weight
    if original.head.bias is not None:
        replacement.head.bias = original.head.bias

    return replacement


def InitEpsilonLinearLRP(original):
    in_features = original.in_features
    out_features = original.out_features

    replacement = EpsilonLinearLRP(in_features, out_features, bias=False)
    replacement.weight = original.weight
    if original.bias is not None:
        replacement.bias = original.bias
    return replacement


def InitBatchNorm2dLRP(original):
    kwargs = {}
    for arg in inspect.signature(original.__init__).parameters.keys():
        if hasattr(original, arg):
            kwargs[arg] = getattr(original, arg)

    replacement = BatchNorm2dLRP(**kwargs)
    replacement.load_state_dict(original.state_dict())

    return replacement



