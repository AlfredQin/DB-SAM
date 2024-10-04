from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from src.models.ops.ms_deform_attn.functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_query=256, d_value=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_query % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_query, n_heads))
        _d_per_head = d_query // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_query = d_query
        self.d_value = d_value
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_query, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_query, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_value, d_value)
        self.output_proj = nn.Linear(d_value, d_value)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                              self.n_levels,
                                                                                                              self.n_points,
                                                                                                              1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
            self.im2col_step)
        output = self.output_proj(output)
        return output


class MultiResMultiFreqDeformAttnBase(nn.Module):
    def __init__(
            self,
            d_model=768,
            n_levels=2,
            n_freqs=4,
            n_heads=24,
            n_points=4,
    ):
        super().__init__()
        """
        Multi-Scale Deformable Attention Module
        :param d_model          hidden dimension
        :param n_levels         number of feature levels
        :param n_heads          number of attention heads
        :param n_curr_points    number of sampling points per attention head per feature level from
                                each query corresponding frame
        :param n_temporal_points    number of sampling points per attention head per feature level
                                    from temporal frames
        """
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_freqs = n_freqs
        self.n_heads = n_heads
        self.n_points = n_points

        # Used for sampling and attention in the prev or post frames
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_freqs * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_freqs * n_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all
        constant_(self.temporal_sampling_offsets.weight.data, 0.)

        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)

        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])

        # curr_frame init
        curr_grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels,
                                                                      self.n_curr_frame_points, 1)
        for i in range(self.n_curr_points):
            curr_grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(curr_grid_init.reshape(-1))

        # other_frames init
        temporal_grid_init = grid_init.view(self.n_heads, 1, 1, 1, 2).repeat(1, self.n_levels,
                                                                             self.n_other_frames,
                                                                             self.n_other_frame_points,
                                                                             1)

        for i in range(self.n_other_frame_points):
            temporal_grid_init[:, :, :, i, :] *= i + 1

        with torch.no_grad():
            self.temporal_sampling_offsets.bias = nn.Parameter(temporal_grid_init.reshape(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        constant_(self.temporal_attention_weights.weight.data, 0.)
        constant_(self.temporal_attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask
    ):
        """
        Args:
            query:
            reference_points:
            input_flatten:
            input_spatial_shapes:
            input_level_start_index:
            input_padding_mask:

        Returns:

        """
        raise NotImplementedError

    # Computes current/temporal sampling offsets and attention weights,
    # which are treated different for the encoder and decoder later on
    def _compute_deformable_attention(self, query, input_flatten):
        B, Len_q, _ = query.shape
        value = self.value_proj(input_flatten)
        value = value.view(B, -1, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_freqs, self.n_levels, self.n_points, 2
        ).flatten(3, 4)
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_freqs * self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = attention_weights.view(
            B, Len_q, self.n_heads, self.n_freqs * self.n_levels, self.n_points
        ).contiguous()

        return value, sampling_offsets, attention_weights


class MultiResMultiFreqDeformCrossAttn(MultiResMultiFreqDeformAttnBase):
    def __init__(
            self,
            d_model=768,
            n_levels=2,
            n_freqs=4,
            n_heads=24,
            n_points=4,
    ):
        super().__init__(d_model, n_levels, n_freqs, n_heads, n_points)

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask=None
    ):
        """
        Args:
            query: torch.tensor([B, Len_q, D])
            reference_points: torch.tensor([B, Len_q, 2, 2])
            input_flatten: torch.tensor([B, N_freq, Len_in, D]), Len_in = (sum(Hi * Wi) for i in levels)
            input_spatial_shapes: torch.tensor([N_lvl, 2])
            input_level_start_index: [0, 64, 96, ...]
            input_padding_mask: [B, Len_in]
        """
        B, Len_q, _ = query.shape
        B, N_freq, Len_in, _ = input_flatten.shape
        value, sampling_offsets, attention_weights = self._compute_deformable_attention(query, input_flatten)
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        offset_normalizer = offset_normalizer.repeat(self.n_freqs, 1)  # [N_freq * N_lvl, 2]

        reference_points = reference_points[:, :, 0, :][:, :, None, None, None]  # [B, Len_q, 1, 1, 1, 2]
        sampling_locations = reference_points + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        input_spatial_shapes = input_spatial_shapes.repeat(self.n_freqs, 1)  # [N_freq * N_lvl, 2]
        input_level_start_index = torch.cat(
            [input_spatial_shapes.new_zeros(1, ), input_spatial_shapes.prod(1).cumsum(0)[:-1]])
        out = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
            self.im2col_step
        )
        out = self.output_proj(out)

        return out


class FRANMsDeformAttn(MSDeformAttn):
    def __init__(self, d_query=256, d_value=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__(d_query, d_value, n_levels, n_heads, n_points)
        self.gamma_list = nn.ModuleList([nn.Conv2d(d_query, d_query, kernel_size=3, stride=1, padding=1, bias=True)
                                         for _ in range(n_levels - 1)])
        self.beta_list = nn.ModuleList([nn.Conv2d(d_query, d_query, kernel_size=3, stride=1, padding=1, bias=True)
                                        for _ in range(n_levels - 1)])

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()