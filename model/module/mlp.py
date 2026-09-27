import torch.nn as nn
from timm.layers import to_2tuple

class Mlp(nn.Module):
    def __init__(self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features,bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features,bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def get_complexity(self, tokens_len, dim_rate, hidden_rate):
        total_flops = 0
        # fc1 FLOPs:
        total_flops += tokens_len * self.in_features * self.hidden_features * (1 - dim_rate) * (1 - hidden_rate)
        total_flops += tokens_len * self.out_features * self.hidden_features * (1 - dim_rate) * (1 - hidden_rate)
        return total_flops



