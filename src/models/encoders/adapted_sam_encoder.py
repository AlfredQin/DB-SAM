import torch
import torch.nn as nn
from copy import deepcopy

from typing import Optional, Tuple, Type, Union

from super_gradients.common.decorators.factory_decorator import resolve_param

from .registry import register_encoder
from src.models.segmentation_models.segment_anything.modeling.common import LayerNorm2d
from src.models.segmentation_models.segment_anything.modeling.image_encoder import PatchEmbed, Block
from src.models.adapters.registry import AdapterFactory


@register_encoder()
class AdaptedViTEncoder(nn.Module):
    @resolve_param("adapter", factory=AdapterFactory())
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        adapter: Union[nn.Module, dict] = None,
        pretrained_sam_encoder_weights=None,
        output_ms_feats: bool = True,
        finetune_neck: bool = True
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            finetune_neck: If the necks.requires_grad is True
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
            )
        self.load_state_dict(pretrained_sam_encoder_weights, strict=True)

        self.adapter = adapter

        # freeze parameters of encoder
        for name, parameter in self.named_parameters():
            if "adapter" not in name:
                parameter.requires_grad = False
        if finetune_neck:
            for name, parameter in self.neck.named_parameters():
                parameter.requires_grad = True

        self.output_ms_feats = output_ms_feats

    def forward(self, input_image):
        out = self.adapter(input_image, self)
        multi_scale_feats = out["multi_scale_feats"]
        if not self.output_ms_feats:
            multi_scale_feats = multi_scale_feats[-2]

        return {"multi_scale_feats": multi_scale_feats, "conv_feats": out["conv_feats"]}
