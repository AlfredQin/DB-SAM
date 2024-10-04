import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
import copy

from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from src.models.ops.ms_deform_attn.modules.ms_deform_attn import MSDeformAttn
from ..common.conv_layer import MbConvBlock



class BaseConvViTAdapter(nn.Module):
    def __init__(self, ):
        super().__init__()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, input_image, vit_encoder):
        raise NotImplementedError


class FusionBlock(nn.Module):
    def __init__(self, dim=768, norm_layer=LayerNorm2d):
        super().__init__()
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.fusion = MbConvBlock(dim, dim)

    def forward(self, feat0, feat1):
        """
        :param feat0: B, C, H, W
        :param feat1: B, C, H, W
        :return: feat: B, C, H, W
        """
        feat = self.norm0(feat0) + self.norm1(feat1)
        feat = self.fusion(feat)

        return feat
