import torch
import torch.nn as nn
from .registry import register_spatial_prior_module
from torchvision.transforms import Resize
from super_gradients.training.models import BasicResNetBlock
from timm.layers import LayerNorm2d

class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.SyncBatchNorm(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Iter_Downsample(nn.Module):

    def __init__(self, ):
        super(Iter_Downsample, self).__init__()
        self.init_ds = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.init_ds(x)
        x1 = self.ds1(x)
        x2 = self.ds2(x1)
        x3 = self.ds3(x2)
        x4 = self.ds4(x3)
        return x1, x2, x3, x4


@register_spatial_prior_module()
class LFIP(nn.Module):
    def __init__(self, in_planes, embed_dim: int = 768, ms_feats_levels_index: tuple = (3, 4)):
        super().__init__()
        self.iter_ds = Iter_Downsample1()
        lcbs = dict(
            lcb1=nn.Sequential(
                ConvBlock(in_planes=in_planes, out_planes=256, kernel_size=3, padding=1, stride=2),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=3, padding=1, ),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=3, padding=1, ),
                ConvBlock(in_planes=256, out_planes=embed_dim, kernel_size=1, relu=False, ),
            ),
            lcb2=nn.Sequential(
                ConvBlock(in_planes=in_planes, out_planes=256, kernel_size=3, padding=1, stride=2),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=3, padding=1, ),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=3, padding=1, ),
                ConvBlock(in_planes=256, out_planes=embed_dim, kernel_size=1, relu=False, ),
            ),
            lcb3=nn.Sequential(
                ConvBlock(in_planes=in_planes, out_planes=192, kernel_size=3, padding=1, stride=2),
                ConvBlock(in_planes=192, out_planes=192, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=192, out_planes=192, kernel_size=3, padding=1, ),
                ConvBlock(in_planes=192, out_planes=192, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=192, out_planes=192, kernel_size=3, padding=1, ),
                ConvBlock(in_planes=192, out_planes=embed_dim, kernel_size=1, relu=False),
            ),
            lcb4=nn.Sequential(
                ConvBlock(in_planes=in_planes, out_planes=192, kernel_size=3, padding=1, stride=2),
                ConvBlock(in_planes=192, out_planes=192, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=192, out_planes=192, kernel_size=3, padding=1, ),
                ConvBlock(in_planes=192, out_planes=192, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=192, out_planes=192, kernel_size=3, padding=1, ),
                ConvBlock(in_planes=192, out_planes=embed_dim, kernel_size=1, relu=False, ),
            )
        )
        # adopt lcb according the number in the ms_feats_levels_index
        for lvl_idx in ms_feats_levels_index:
            self.__setattr__(name=f"lcb{lvl_idx}", value=lcbs[f"lcb{lvl_idx}"])
        self.ms_feats_levels_index = ms_feats_levels_index

    def forward(self, x):
        down_sample_images = self.iter_ds(x)
        ms_feats = {}
        for lvl_idx in self.ms_feats_levels_index:
            x = self.__getattr__(f"lcb{lvl_idx}")(down_sample_images[lvl_idx - 1])
            x = x.flatten(2).transpose(1, 2)  # B, N, C
            ms_feats[f"c{lvl_idx}"] = x

        return ms_feats

class Iter_Downsample1(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ds1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x1 = self.ds1(x)
        x2 = self.ds2(x1)
        x3 = self.ds3(x2)
        x4 = self.ds4(x3)
        return x1, x2, x3, x4



@register_spatial_prior_module()
class LFIP6(nn.Module):
    """
    large conv kernel
    """
    def __init__(self, in_planes, embed_dim: int = 768, ms_feats_levels_index: tuple = (3, 4)):
        super().__init__()
        lcbs = dict(
            lcb1=nn.Sequential(
                ConvBlock(in_planes=in_planes, out_planes=128, kernel_size=7, stride=4, padding=1),
                ConvBlock(in_planes=128, out_planes=256, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=512, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=512, out_planes=512, kernel_size=1, padding=0, ),
                ConvBlock(in_planes=512, out_planes=embed_dim, kernel_size=1, relu=False, bn=False),
            ),
            lcb2=nn.Sequential(
                Resize(512),
                ConvBlock(in_planes=in_planes, out_planes=128, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=128, out_planes=256, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=512, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=512, out_planes=512, kernel_size=1, padding=0, ),
                ConvBlock(in_planes=512, out_planes=embed_dim, kernel_size=1, relu=False, bn=False),
            ),
            lcb3=nn.Sequential(
                Resize(256),
                ConvBlock(in_planes=in_planes, out_planes=256, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=512, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=512, out_planes=512, kernel_size=1, padding=0, ),
                ConvBlock(in_planes=512, out_planes=embed_dim, kernel_size=1, relu=False, bn=False),
            ),
            lcb4=nn.Sequential(
                Resize(128),
                ConvBlock(in_planes=in_planes, out_planes=256, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=512, kernel_size=3, stride=1, padding=1),
                ConvBlock(in_planes=512, out_planes=512, kernel_size=1, padding=0, ),
                ConvBlock(in_planes=512, out_planes=embed_dim, kernel_size=1, relu=False, bn=False),
            ),
        )
        # adopt lcb according the number in the ms_feats_levels_index
        for lvl_idx in ms_feats_levels_index:
            self.__setattr__(name=f"lcb{lvl_idx}", value=lcbs[f"lcb{lvl_idx}"])
        self.ms_feats_levels_index = ms_feats_levels_index

    def forward(self, x):
        down_sample_images = [x, x, x, x,]
        ms_feats = {}
        for lvl_idx in self.ms_feats_levels_index:
            x = self.__getattr__(f"lcb{lvl_idx}")(down_sample_images[lvl_idx - 1])
            x = x.permute(0, 2, 3, 1)
            ms_feats[f"c{lvl_idx}"] = x

        return ms_feats



@register_spatial_prior_module()
class LFIP0(LFIP6):
    """
    Small conv kernel
    """
    def __init__(self, in_planes, embed_dim: int = 768, ms_feats_levels_index: tuple = (3, 4), layer_norm=True):
        super(LFIP6, self).__init__()
        lcbs = dict(
            lcb1=nn.Sequential(
                Resize(512),
                ConvBlock(in_planes=in_planes, out_planes=128, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=128, out_planes=128, kernel_size=1, padding=0, ),
                ConvBlock(in_planes=128, out_planes=256, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=512, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=512, out_planes=512, kernel_size=1, padding=0, ),
                ConvBlock(in_planes=512, out_planes=embed_dim, kernel_size=1, relu=False, bn=False),
                LayerNorm2d(embed_dim) if layer_norm else nn.Identity(),
            ),
            lcb2=nn.Sequential(
                Resize(256),
                ConvBlock(in_planes=in_planes, out_planes=128, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=128, out_planes=128, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=128, out_planes=256, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=512, kernel_size=3, stride=1, padding=1),
                ConvBlock(in_planes=512, out_planes=512, kernel_size=1, padding=0, ),
                ConvBlock(in_planes=512, out_planes=embed_dim, kernel_size=1, relu=False, bn=False),
                LayerNorm2d(embed_dim) if layer_norm else nn.Identity(),
            ),
            lcb3=nn.Sequential(
                Resize(128),
                ConvBlock(in_planes=in_planes, out_planes=128, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_planes=128, out_planes=128, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=128, out_planes=256, kernel_size=3, stride=1, padding=1),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1,),
                ConvBlock(in_planes=256, out_planes=512, kernel_size=3, stride=1, padding=1),
                ConvBlock(in_planes=512, out_planes=512, kernel_size=1, padding=0, ),
                ConvBlock(in_planes=512, out_planes=embed_dim, kernel_size=1, relu=False, bn=False),
                LayerNorm2d(embed_dim) if layer_norm else nn.Identity(),
            ),
            lcb4=nn.Sequential(
                Resize(64),
                ConvBlock(in_planes=in_planes, out_planes=128, kernel_size=3, stride=1, padding=1),
                ConvBlock(in_planes=128, out_planes=128, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=128, out_planes=256, kernel_size=3, stride=1, padding=1),
                ConvBlock(in_planes=256, out_planes=256, kernel_size=1, stride=1, ),
                ConvBlock(in_planes=256, out_planes=512, kernel_size=3, stride=1, padding=1),
                ConvBlock(in_planes=512, out_planes=512, kernel_size=1, padding=0, ),
                ConvBlock(in_planes=512, out_planes=embed_dim, kernel_size=1, relu=False, bn=False),
                LayerNorm2d(embed_dim) if layer_norm else nn.Identity(),
            ),
        )
        # adopt lcb according the number in the ms_feats_levels_index
        for lvl_idx in ms_feats_levels_index:
            self.__setattr__(name=f"lcb{lvl_idx}", value=lcbs[f"lcb{lvl_idx}"])
        self.ms_feats_levels_index = ms_feats_levels_index
