from super_gradients.common.registry.registry import create_register_decorator
from super_gradients.common.factories.base_factory import BaseFactory

VIT_EMBEDDING_EXTRACTOR = {}
register_vit_embedding_extractor = create_register_decorator(registry=VIT_EMBEDDING_EXTRACTOR)


class VitEmbeddingExtractorFactory(BaseFactory):
    def __init__(self):
        super().__init__(VIT_EMBEDDING_EXTRACTOR)


VIT_FEAT_CONV_FEAT_FUSION_BLOCKS = {}
register_vit_feat_conv_feat_fusion_blocks = create_register_decorator(registry=VIT_FEAT_CONV_FEAT_FUSION_BLOCKS)


class ViTFeatConvFeatFusionBlocksFactory(BaseFactory):
    def __init__(self):
        super().__init__(VIT_FEAT_CONV_FEAT_FUSION_BLOCKS)


VIT_CONV_INTERACT_BLOCKS = {}
register_vit_conv_interact_blocks = create_register_decorator(registry=VIT_CONV_INTERACT_BLOCKS)


class ViTConvInteractBlocksFactory(BaseFactory):
    def __init__(self):
        super().__init__(VIT_CONV_INTERACT_BLOCKS)


VIT_CONV_AGGREGATION_BLOCKS = {}
register_vit_conv_aggregation_blocks = create_register_decorator(registry=VIT_CONV_AGGREGATION_BLOCKS)


class ViTConvAggregationBlocksFactory(BaseFactory):
    def __init__(self):
        super().__init__(VIT_CONV_AGGREGATION_BLOCKS)

