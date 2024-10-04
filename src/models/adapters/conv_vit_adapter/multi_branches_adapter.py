import torch
import torch.nn as nn
from torch.nn.init import normal_
from typing import Union, Optional, Callable, List, Tuple, Dict

from super_gradients.common.decorators.factory_decorator import resolve_param

from ..registry import register_adapter
from src.models.spatial_prior_modules import SpatialPriorModuleFactory

from ..common import ViTConvAggregationBlocksFactory
from ..common.conv_layer import MbConvBlock
from ..common.interact_block import MSDeformCrossAttnBlock

from .adapter import BaseConvViTAdapter



@register_adapter()
class TwoBranchImagePyramidAdapter(BaseConvViTAdapter):
    @resolve_param("spatial_prior_module", factory=SpatialPriorModuleFactory())
    @resolve_param("aggregation", factory=ViTConvAggregationBlocksFactory())
    def __init__(
            self,
            spatial_prior_module: Union[nn.Module, dict],
            vit_embed_dim=768,
            ms_feats_levels_index: Tuple = (2, 3),
            interaction_indexes=None,
            using_shallow_feats: bool = True,
            attn_drop=0.1,
            proj_drop=0.1,
            drop_path_attn=0.3,
            drop_path_ffn=0.3,
            drop_path_conv=0.3,
            num_heads=12,
            expand_ratio=4,
            vit_conv_position=None,
            aggregation=None,
    ):
        """
        :param spatial_prior_module: spatial prior module
        :param vit_embed_dim: vit embedding dimension
        :param ms_feats_levels_index: multi-scale features levels index, for example, (2, 3) means using c2 and c3
        """
        super().__init__()
        if using_shallow_feats:
            self.spatial_prior_module = spatial_prior_module
        self.using_shallow_feats = using_shallow_feats
        self.num_stages = len(interaction_indexes)
        self.num_ms_feats_levels = len(ms_feats_levels_index)
        self.ms_feats_levels_index = ms_feats_levels_index

        self.vit_cross_attn_blocks = nn.ModuleList([
            MSDeformCrossAttnBlock(dim=vit_embed_dim, num_heads=num_heads, n_levels=len(ms_feats_levels_index),
                                   n_points=4, attn_drop=attn_drop,
                                   proj_drop=proj_drop, drop_path_attn=drop_path_attn, drop_path_ffn=drop_path_ffn,
                                   expand_ratio=expand_ratio, query_norm=True, ffn_pre_norm=True,)
            for _ in range(self.num_stages)
        ])
        self.conv_cross_attn_blocks = nn.ModuleList([
            MSDeformCrossAttnBlock(dim=vit_embed_dim, num_heads=num_heads, n_levels=1, n_points=4, attn_drop=attn_drop,
                                   proj_drop=proj_drop, drop_path_attn=drop_path_attn, drop_path_ffn=drop_path_ffn,
                                   expand_ratio=expand_ratio, query_norm=True, ffn_pre_norm=True,)
            for _ in range(self.num_stages)
        ])

        self.vit_conv_blocks = nn.ModuleList([
            MbConvBlock(vit_embed_dim, vit_embed_dim, expand_ratio=1, drop_path_rate=drop_path_conv)
            for _ in range(self.num_stages)])

        self.vit_conv_position = vit_conv_position
        self.interaction_indexes = interaction_indexes
        self.aggregation = aggregation

        self.norm = nn.LayerNorm(vit_embed_dim)
        self.level_embed = nn.Parameter(torch.zeros(self.num_ms_feats_levels, vit_embed_dim))
        self.conv = nn.Conv2d(vit_embed_dim * len(ms_feats_levels_index), vit_embed_dim, kernel_size=1, bias=True)

        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def forward(self, input_image, vit_encoder):
        vit_embedding = vit_encoder.patch_embed(input_image)  # B, 64x64x768
        if vit_encoder.pos_embed is not None:
            vit_embedding = vit_embedding + vit_encoder.pos_embed
        bs, vit_h, vit_w, dim = vit_embedding.shape  # B H W C
        shallow_feats = self.spatial_prior_module(input_image)
        ms_feats = []
        scale_list = [f"c{i}" for i in self.ms_feats_levels_index]
        for idx, scale in enumerate(scale_list):
            shallow_feat = shallow_feats[scale] + self.level_embed[idx].view(1, 1, 1, -1)
            ms_feats.append(shallow_feat)
        ms_feats = [ms_feat.permute(0, 3, 1, 2) for ms_feat in ms_feats]

        intermediate_ms_feats = []
        for stage_i, index in enumerate(self.interaction_indexes):
            for blk_i, vit_blk in enumerate(vit_encoder.blocks[index[0]:index[-1] + 1]):
                vit_embedding = vit_blk(vit_embedding)
            if self.vit_conv_position == "before":
                vit_embedding = self.vit_conv_blocks[stage_i](vit_embedding.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            vit_embedding_ = [vit_embedding.permute(0, 3, 1, 2)]
            vit_embedding = self.vit_cross_attn_blocks[stage_i](vit_embedding_, ms_feats)[0].permute(0, 2, 3, 1)
            ms_feats = self.conv_cross_attn_blocks[stage_i](ms_feats, vit_embedding_)

            if self.vit_conv_position == "after":
                vit_embedding = self.vit_conv_blocks[stage_i](vit_embedding.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        conv_feat = self.conv(torch.cat(ms_feats, dim=1)).permute(0, 2, 3, 1)
        vit_embedding = self.aggregation(vit_embedding, conv_feat)

        output = vit_encoder.neck(
            self.norm(vit_embedding.flatten(1, 2)).view(-1, vit_h, vit_w, dim).permute(0, 3, 1, 2)
        )
        outputs = [None, None, output, None]

        return {
            "multi_scale_feats": outputs,
            "conv_feats": intermediate_ms_feats
        }


@register_adapter()
class FourBranchImagePyramidAdapter(BaseConvViTAdapter):
    @resolve_param("spatial_prior_module", factory=SpatialPriorModuleFactory())
    @resolve_param("aggregation", factory=ViTConvAggregationBlocksFactory())
    def __init__(
            self,
            spatial_prior_module: Union[nn.Module, dict],
            vit_embed_dim=768,
            ms_feats_levels_index: Tuple = (2, 3),
            interaction_indexes=None,
            using_shallow_feats: bool = True,
            attn_drop=0.1,
            proj_drop=0.1,
            drop_path_attn=0.3,
            drop_path_ffn=0.3,
            drop_path_conv=0.3,
            num_heads=12,
            expand_ratio=4,
            vit_conv_position=None,
            aggregation=None,
    ):
        """
        :param spatial_prior_module: spatial prior module
        :param vit_embed_dim: vit embedding dimension
        :param ms_feats_levels_index: multi-scale features levels index, for example, (2, 3) means using c2 and c3
        """
        super().__init__()
        if using_shallow_feats:
            self.spatial_prior_module = spatial_prior_module
        self.using_shallow_feats = using_shallow_feats
        self.num_stages = len(interaction_indexes)
        self.num_ms_feats_levels = len(ms_feats_levels_index)
        self.ms_feats_levels_index = ms_feats_levels_index

        self.self_attn_blocks = nn.ModuleList([
            MSDeformCrossAttnBlock(dim=vit_embed_dim, num_heads=num_heads, n_levels=len(ms_feats_levels_index) + 1,
                                   n_points=4, attn_drop=attn_drop,
                                   proj_drop=proj_drop, drop_path_attn=drop_path_attn, drop_path_ffn=drop_path_ffn,
                                   expand_ratio=expand_ratio, query_norm=True, ffn_pre_norm=True,)
            for _ in range(self.num_stages)
        ])
        self.self_attn_blocks1 = nn.ModuleList([
            MSDeformCrossAttnBlock(dim=vit_embed_dim, num_heads=num_heads, n_levels=len(ms_feats_levels_index) + 1,
                                   n_points=4, attn_drop=attn_drop,
                                   proj_drop=proj_drop, drop_path_attn=drop_path_attn, drop_path_ffn=drop_path_ffn,
                                   expand_ratio=expand_ratio, query_norm=True, ffn_pre_norm=True,)
            for _ in range(self.num_stages)
        ])
        self.interaction_indexes = interaction_indexes
        self.aggregation = aggregation

        self.norm = nn.LayerNorm(vit_embed_dim)
        self.level_embed = nn.Parameter(torch.zeros(self.num_ms_feats_levels, vit_embed_dim))

        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def forward(self, input_image, vit_encoder):
        vit_embedding = vit_encoder.patch_embed(input_image)  # B, 64x64x768
        if vit_encoder.pos_embed is not None:
            vit_embedding = vit_embedding + vit_encoder.pos_embed
        bs, vit_h, vit_w, dim = vit_embedding.shape  # B H W C
        shallow_feats = self.spatial_prior_module(input_image)
        ms_feats = []
        scale_list = [f"c{i}" for i in self.ms_feats_levels_index]
        for idx, scale in enumerate(scale_list):
            shallow_feat = shallow_feats[scale] + self.level_embed[idx].view(1, 1, 1, -1)
            ms_feats.append(shallow_feat)
        ms_feats = [ms_feat.permute(0, 3, 1, 2) for ms_feat in ms_feats]

        intermediate_ms_feats = []
        for stage_i, index in enumerate(self.interaction_indexes):
            for blk_i, vit_blk in enumerate(vit_encoder.blocks[index[0]:index[-1] + 1]):
                vit_embedding = vit_blk(vit_embedding)

            feats_pyramid = [vit_embedding.permute(0, 3, 1, 2)] + ms_feats
            feats_pyramid = self.self_attn_blocks[stage_i](feats_pyramid, feats_pyramid)
            feats_pyramid = self.self_attn_blocks1[stage_i](feats_pyramid, feats_pyramid)
            vit_embedding = feats_pyramid[0].permute(0, 2, 3, 1)
            ms_feats = feats_pyramid[1:]

        conv_feat = [f.permute(0, 2, 3, 1) for f in ms_feats]
        vit_embedding = self.aggregation(vit_embedding, conv_feat)

        output = vit_encoder.neck(
            self.norm(vit_embedding.flatten(1, 2)).view(-1, vit_h, vit_w, dim).permute(0, 3, 1, 2)
        )
        outputs = [None, None, output, None]

        return {
            "multi_scale_feats": outputs,
            "conv_feats": intermediate_ms_feats
        }

