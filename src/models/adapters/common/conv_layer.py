
import torch.nn as nn
from timm.models.layers import get_attn, LayerNorm2d, DropPath


class MbConvBlock(nn.Module):
    """ A depthwise separable / fused mbconv style residual block with SE, `no norm.
    """
    def __init__(
            self,
            in_chs,
            out_chs=None,
            expand_ratio=1.0,
            attn_layer='se',
            bias=False,
            act_layer=nn.GELU,
            norm_layer=LayerNorm2d,
            drop_path_rate=0.0,
    ):
        super().__init__()
        attn_kwargs = dict(act_layer=act_layer)
        if isinstance(attn_layer, str) and attn_layer == 'se' or attn_layer == 'eca':
            attn_kwargs['rd_ratio'] = 0.25
            attn_kwargs['bias'] = False
        attn_layer = get_attn(attn_layer)
        out_chs = out_chs or in_chs
        mid_chs = int(expand_ratio * in_chs)

        self.pre_norm = norm_layer(in_chs) if norm_layer is not None else nn.Identity()
        self.conv_dw = nn.Conv2d(in_chs, mid_chs, 3, 1, 1, groups=in_chs, bias=bias)
        self.act = act_layer()
        self.se = attn_layer(mid_chs, **attn_kwargs)
        self.conv_pw = nn.Conv2d(mid_chs, out_chs, 1, 1, 0, bias=bias)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.pre_norm(x)
        x = self.conv_dw(x)
        x = self.act(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.drop_path(x) + shortcut
        return x