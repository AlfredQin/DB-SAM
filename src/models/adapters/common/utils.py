
import torch
from typing import List

from ..vit_adapter.interaction_block import get_reference_points


def get_deform_inputs(
        high_resolution_feat_shape: tuple = (256, 256),
        num_feats_levels: int = 4,
        query_shape: List[tuple] = (64, 64),
        device: torch.device = torch.device("cpu"),
        variable_scales: bool = True,
):
    """
    params: variable_scales: if True, the size of each feature map is different, otherwise, the size of each feature map is the same.
    """
    h, w = high_resolution_feat_shape
    if variable_scales:
        spatial_shapes = torch.as_tensor([[h // (2 ** i), w // (2 ** i)] for i in range(num_feats_levels)],
                                         dtype=torch.long, device=device)
    else:
        spatial_shapes = torch.as_tensor([[h, w] for _ in range(num_feats_levels)], dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )

    if isinstance(query_shape[0], int):  # fix bugs
        query_shape = [query_shape]
    reference_points = get_reference_points(query_shape, device=device)

    return [reference_points, spatial_shapes, level_start_index]