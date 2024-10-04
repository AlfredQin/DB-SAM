import torch
import torch.nn as nn

from timm.models.layers import DropPath
from .utils import get_deform_inputs
from src.models.ops.ms_deform_attn.modules import MSDeformAttn
from src.models.transformer.ms_deform.deformable_encoder import MSMixFFN


class MSDeformCrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, n_levels, n_points, attn_drop=0., proj_drop=0., drop_path_attn=0.,
                 drop_path_ffn=0., expand_ratio=4, query_norm=False, ffn_pre_norm=False,
                 ):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim) if query_norm else nn.Identity()
        self.value_norm = nn.LayerNorm(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points)
        self.dropout_attn = nn.Dropout(attn_drop)
        self.drop_path_attn = DropPath(drop_path_attn) if drop_path_attn > 0. else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_ffn) if drop_path_ffn > 0. else nn.Identity()

        self.ffn_pre_norm = nn.LayerNorm(dim) if ffn_pre_norm else nn.Identity()
        self.mlp = MSMixFFN(dim, hidden_features=int(dim * expand_ratio), act_layer=nn.GELU, drop=proj_drop)

    def forward(self, query, value):
        """
        :param query: List of tensor [B, C, H, W]
        :param value: List of tensor [B, C, H, W]
        """
        query_shapes = [q.shape for q in query]
        deform_inputs = get_deform_inputs(tuple(value[0].shape[2:]), len(value), [tuple(q.shape[2:]) for q in query],
                                          query[0].device, variable_scales=value[0].shape != value[-1].shape)
        query = torch.cat([q.flatten(2).permute(0, 2, 1) for q in query], dim=1)

        x = self.attn(
            query=self.query_norm(query),
            reference_points=deform_inputs[0],
            input_flatten=self.value_norm(torch.cat([v.flatten(2).permute(0, 2, 1) for v in value], dim=1)),
            input_spatial_shapes=deform_inputs[1],
            input_level_start_index=deform_inputs[2],
        )
        x = query + self.drop_path_attn(self.dropout_attn(x))
        x = x + self.drop_path_ffn(self.mlp(self.ffn_pre_norm(x), [shape[-2:] for shape in query_shapes]))

        # split the output of MSDeformAttn into different levels
        x = torch.split(x, [shape[-2] * shape[-1] for shape in query_shapes], dim=1)
        x = [q.permute(0, 2, 1).view(*shape) for q, shape in zip(x, query_shapes)]

        return x
