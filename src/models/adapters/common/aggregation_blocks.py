
import torch
import torch.nn as nn
from timm.models.layers import DropPath, LayerNorm2d, Mlp
from .conv_layer import MbConvBlock
from .registry import register_vit_conv_aggregation_blocks


@register_vit_conv_aggregation_blocks()
class ConcatAggregationBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=4.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0),
            MbConvBlock(dim, dim, expand_ratio=expansion_ratio)
        )

    def forward(self, vit_feat, conv_feat):
        """
        :param vit_feat: [B, H, W, C]
        :param conv_feat: [B, H, W, C]
        """
        out = self.conv(torch.cat([vit_feat, conv_feat], dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out


@register_vit_conv_aggregation_blocks()
class DANE(nn.Module):
    def __init__(
            self,
            dim,
            reduction=12,
            drop_path=0.3,
            ffn_type: str = "mlp",
            expand_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = dim
        self.fc_spatial = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_channel = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(dim // reduction),
            nn.Linear(dim // reduction, dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        if ffn_type == "mlp":
            self.ffn_norm = nn.LayerNorm(dim)
            self.ffn = Mlp(dim, int(expand_ratio * dim), dim)

        self.ffn_type = ffn_type
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_tf, x_cnn):
        B, H, W, C = x_cnn.shape
        x_tf = x_tf.view(B, -1, C)
        x_cnn = x_cnn.view(B, -1, C)
        # B L C
        x_spatial_mask = self.fc_spatial(x_cnn)  # B L 1
        x_channel_mask = self.fc_channel(self.avg_pool(x_tf.permute(0, 2, 1)).permute(0, 2, 1))  # B 1 C
        x_mask = self.sigmoid(x_spatial_mask.expand_as(x_cnn) + x_channel_mask.expand_as(x_tf))
        out = x_cnn * x_mask + x_tf * (1 - x_mask)

        if self.ffn_type == "mlp":
            out = out + self.drop_path(self.ffn(self.ffn_norm(out)))
        out = out.view(B, H, W, C)

        return out


@register_vit_conv_aggregation_blocks()
class HadamardProduct(nn.Module):
    def __init__(
            self,
            dim=768,
    ):
        super().__init__()
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.prob = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

    def forward(self, vit_feat, conv_feat):
        """
        :param vit_feat: [B, H, W, C]
        :param conv_feat: [B, H, W, C]
        """
        prob = self.prob(
            self.to_q(vit_feat.permute(0, 3, 1, 2)) * self.to_k(conv_feat.permute(0, 3, 1, 2))
        )
        out = prob * vit_feat.permute(0, 3, 1, 2) + (1 - prob) * conv_feat.permute(0, 3, 1, 2)
        out = out.permute(0, 2, 3, 1)

        return out


@register_vit_conv_aggregation_blocks()
class Add(nn.Module):
    def __init__(
            self,
            *args, **kwargs
    ):
        super().__init__()

    def forward(self, vit_feat, conv_feat):
        """
        :param vit_feat: [B, H, W, C]
        :param conv_feat: [B, H, W, C]
        """
        out = vit_feat + conv_feat

        return out


@register_vit_conv_aggregation_blocks()
class Select(nn.Module):
    def __init__(
            self,
            name: str = "vit",
            **kwargs
    ):
        super().__init__()
        self.name = name

    def forward(self, vit_feat, conv_feat):
        """
        :param vit_feat: [B, H, W, C]
        :param conv_feat: [B, H, W, C]
        """
        if self.name == "vit":
            out = vit_feat
        else:
            out = conv_feat

        return out


@register_vit_conv_aggregation_blocks()
class DANE2(nn.Module):
    def __init__(
            self,
            dim,
            reduction=12,
            drop_path=0.3,
            ffn_type: str = "mlp",
            expand_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = dim
        self.fc_spatial = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_channel = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(dim // reduction),
            nn.Linear(dim // reduction, dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        self.fc_spatial1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1, bias=False),
        )
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)
        self.fc_channel1 = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(dim // reduction),
            nn.Linear(dim // reduction, dim, bias=False),
        )
        self.sigmoid1 = nn.Sigmoid()

        if ffn_type == "mlp":
            self.ffn_norm = nn.LayerNorm(dim)
            self.ffn = Mlp(dim, int(expand_ratio * dim), dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.ffn_norm1 = nn.LayerNorm(dim)
            self.ffn1 = Mlp(dim, int(expand_ratio * dim), dim)
            self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_type = ffn_type

    def forward(self, x_tf, x_cnn):
        B, H, W, C = x_cnn.shape
        x_tf = x_tf.view(B, -1, C)
        x_cnn = x_cnn.view(B, -1, C)
        # B L C
        x_spatial_mask = self.fc_spatial(x_cnn)  # B L 1
        x_channel_mask = self.fc_channel(self.avg_pool(x_tf.permute(0, 2, 1)).permute(0, 2, 1))  # B 1 C
        x_mask = self.sigmoid(x_spatial_mask.expand_as(x_cnn) + x_channel_mask.expand_as(x_tf))
        out_tf = x_cnn * x_mask + x_tf * (1 - x_mask)

        x_spatial_mask1 = self.fc_spatial1(x_tf)  # B L 1
        x_channel_mask1 = self.fc_channel1(self.avg_pool1(x_cnn.permute(0, 2, 1)).permute(0, 2, 1))  # B 1 C
        x_mask1 = self.sigmoid1(x_spatial_mask1.expand_as(x_tf) + x_channel_mask1.expand_as(x_cnn))
        out_cnn = x_tf * x_mask1 + x_cnn * (1 - x_mask1)

        if self.ffn_type == "mlp":
            out_tf = out_tf + self.drop_path(self.ffn(self.ffn_norm(out_tf)))
            out_cnn = out_cnn + self.drop_path1(self.ffn1(self.ffn_norm1(out_cnn)))

        out_tf = out_tf.view(B, H, W, C)
        out_cnn = out_cnn.view(B, H, W, C)

        return out_tf, out_cnn


@register_vit_conv_aggregation_blocks()
class DANE3(nn.Module):
    def __init__(
            self,
            dim,
            reduction=12,
            drop_path=0.3,
            ffn_type: str = "mlp",
            expand_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = dim
        self.fc_spatial = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(dim // reduction),
            nn.Linear(dim // reduction, dim, bias=False),
        )
        self.fc_channel = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(dim // reduction),
            nn.Linear(dim // reduction, dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        if ffn_type == "mlp":
            self.ffn_norm = nn.LayerNorm(dim)
            self.ffn = Mlp(dim, int(expand_ratio * dim), dim)

        self.ffn_type = ffn_type
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_tf, x_cnn):
        B, H, W, C = x_cnn.shape
        x_tf = x_tf.view(B, -1, C)
        x_cnn = x_cnn.view(B, -1, C)
        # B L C
        x_spatial_mask = self.fc_spatial(x_cnn)  # B L C
        x_channel_mask = self.fc_channel(x_tf)  # B L C
        x_mask = self.sigmoid(x_spatial_mask + x_channel_mask)
        out = x_cnn * x_mask + x_tf * (1 - x_mask)

        if self.ffn_type == "mlp":
            out = out + self.drop_path(self.ffn(self.ffn_norm(out)))
        out = out.view(B, H, W, C)

        return out


@register_vit_conv_aggregation_blocks()
class MultiConvFeatsDANE(nn.Module):
    def __init__(
            self,
            dim,
            num_conv=3,
            reduction=12,
            drop_path=0.3,
            ffn_type: str = "mlp",
            expand_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = dim
        self.vit_fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(dim // reduction),
            nn.Linear(dim // reduction, dim, bias=False),
        )
        self.conv_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // reduction, bias=False),
                nn.SiLU(inplace=True),
                nn.LayerNorm(dim // reduction),
                nn.Linear(dim // reduction, dim, bias=False),
            ) for _ in range(num_conv)
        ])
        self.sigmoid = nn.Softmax()

        if ffn_type == "mlp":
            self.ffn_norm = nn.LayerNorm(dim)
            self.ffn = Mlp(dim, int(expand_ratio * dim), dim)

        self.ffn_type = ffn_type
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_vit, x_cnn):
        """
        :param x_vit: B H W C
        :param x_cnn: List of B H W C
        """
        B, H, W, C = x_vit.shape
        x_vit = x_vit.view(B, -1, C)
        x_cnn = [x.view(B, -1, C) for x in x_cnn]
        # B L C
        conv_mask = [self.conv_fc[i](x_cnn[i]) for i in range(len(x_cnn))]
        vit_mask = self.vit_fc(x_vit)  # B L C
        conv_mask = [m.view(B, H, W, C).permute(0, 3, 1, 2) for m in conv_mask]  # B C H W
        vit_mask = vit_mask.view(B, H, W, C).permute(0, 3, 1, 2)  # B C H W
        stacked_mask = torch.stack(conv_mask + [vit_mask], dim=2)  # B C L H W
        stacked_mask = stacked_mask.permute(0, 1, 3, 4, 2)  # B C H W L
        x_mask = torch.nn.functional.softmax(stacked_mask, dim=-1)  # B C H W L
        x = torch.stack(x_cnn + [x_vit], dim=2)  # B H*W L C
        x = x.view(B, H, W, -1, C).permute(0, 4, 1, 2, 3)  # B C H W L
        out = torch.sum(x * x_mask, dim=-1)  # B C H W
        out = out.flatten(2).transpose(1, 2)  # B H*W C

        if self.ffn_type == "mlp":
            out = out + self.drop_path(self.ffn(self.ffn_norm(out)))
        out = out.view(B, H, W, C)

        return out
