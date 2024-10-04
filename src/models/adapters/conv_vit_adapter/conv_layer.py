import torch
import torch.nn as nn
from timm.layers import LayerNorm2d, Mlp, DropPath
from timm.models.gcvit import MbConvBlock
from typing import List


class ResidualConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.GELU, groups=1,
                 norm_layer=LayerNorm2d, drop_block=None, drop_path=None):
        super().__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer()

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


if __name__ == "__main__":
    torch.cuda.set_device(1)
    state_dict = torch.load("/home/qinc/Code/Segmentation/MedicalSAM/checkpoints/gc_vit_adapter_v6_deform_attn_lfip_sam_mask_decoder_3scale/ckpt_epoch_6.pth")
    net_dict = state_dict["net"]
    # model = make_hr_net_fuse_layers(2, [768, 768 * 2])
    # dane = DANE(768, reduction=12).cuda()
    # conv_feat = torch.randn((1, 32, 32, 768)).view(1, -1, 768).cuda()
    # vit_feat = torch.randn((1, 768, 32, 32)).view(1, -1, 768).cuda()
    # out = dane(vit_feat, conv_feat)
    print()
